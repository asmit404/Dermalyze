import React, { useState, useEffect, useRef } from 'react';
import Button from './ui/Button';
import ResultCard from './ui/ResultCard';
import ProbabilityChart from './ui/ProbabilityChart';
import MedicalInfoCard from './ui/MedicalInfoCard';
import { classInfoMap } from '../lib/classInfo';
import { supabase } from '../lib/supabase';
import { optimizeImage } from '../lib/imageOptimization';
import { encryptImage, getEncryptedExtension } from '../lib/imageEncryption';
import { useDataCache } from '../lib/dataCache';
import { generateDermatologyReport } from '../lib/pdfReportGenerator';
import type { ClassResult, InferenceMetadata } from '../lib/types';

interface ResultsScreenProps {
  image: string | null;
  results: ClassResult[] | null;
  gradcamImage?: string | null;
  caseId: string;
  metadata?: InferenceMetadata | null;
  onAnalyzeAnother: () => void;
  onNavigateToHistory: () => void;
}

/**
 * ResultsScreen — Primary analysis output view.
 *
 * Redesigned following Laws of UX:
 *
 * 1. Jakob's Law — Layout mirrors EHR / radiology PACS conventions:
 *    header with patient/case metadata, prominent diagnosis, two-column
 *    detail view, and a subdued clinical disclaimer footer.
 *
 * 2. Visual Hierarchy — The predicted class name is the largest text on
 *    screen, followed by the confidence score. Secondary information
 *    (probability distribution, image, medical info) is subordinate.
 *
 * 3. Hick's Law — Probability bars are sorted descending so the eye
 *    lands on the dominant prediction immediately. Action buttons are
 *    reduced to two clear choices.
 *
 * 4. Miller's Law — 7 classes (within 7±2 limit). Medical info is
 *    chunked into 4 labeled sections instead of free-form text.
 *
 * 5. Aesthetic-Usability Effect — Neutral medical palette (white, slate,
 *    subtle teal accent), generous spacing, and clean card borders evoke
 *    trust and professionalism.
 *
 * 6. Law of Proximity — Related data (confidence + risk badge) are
 *    adjacent. Image and medical info are in the same column.
 *
 * 7. Law of Common Region — Each logical group lives in its own bordered
 *    card, making the interface instantly scannable.
 *
 * 8. Doherty Threshold — Skeleton loading state provides < 400ms
 *    perceived feedback. Bar animations give immediate visual response.
 *
 * 9. Tesler's Law — Irrelevant metadata removed. Complexity that must
 *    exist (7-class distribution) is presented in the simplest form.
 *
 * 10. Fitts's Law — Primary CTA ("Analyze Another") is full-width with
 *     generous padding. Back button in header is large tap-target.
 */
const ResultsScreen: React.FC<ResultsScreenProps> = ({
  image,
  results,
  gradcamImage,
  caseId,
  metadata,
  onAnalyzeAnother,
  onNavigateToHistory,
}) => {
  const { invalidateAll } = useDataCache();
  const [loading, setLoading] = useState(true);
  const [saveFailed, setSaveFailed] = useState(false);
  const [saveCompleted, setSaveCompleted] = useState(false);
  const [noteText, setNoteText] = useState('');
  const [savingNote, setSavingNote] = useState(false);
  const [noteSaved, setNoteSaved] = useState(false);
  const [noteError, setNoteError] = useState<string | null>(null);
  const [clinicianName, setClinicianName] = useState<string>('');
  const [exportingPdf, setExportingPdf] = useState(false);

  // Doherty Threshold: skeleton -> content in < 400ms
  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 350);
    return () => clearTimeout(timer);
  }, []);

  // Fetch clinician name for PDF report
  useEffect(() => {
    const fetchClinicianName = async () => {
      const {
        data: { user },
      } = await supabase.auth.getUser();
      if (user) {
        // Prefer display name from metadata, fallback to email
        const name =
          user.user_metadata?.full_name || user.user_metadata?.name || user.email || 'Unknown';
        setClinicianName(name);
      }
    };
    fetchClinicianName();
  }, []);

  // Use real results from the API; empty array fallback keeps the UI safe
  const classes: ClassResult[] = results ?? [];

  const predictedClass = classes.length
    ? classes.reduce((prev, cur) => (prev.score > cur.score ? prev : cur))
    : null;

  const info = predictedClass ? classInfoMap[predictedClass.id] : undefined;

  // Case ID is provided by parent component (generated once when results are received)
  // This prevents duplicate records if the component remounts with the same results
  const caseIdDisplay = `DRM-${caseId.slice(0, 8).toUpperCase()}`;
  const analysisDate = new Date().toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });

  // Persist the analysis record + image to Supabase once results are ready
  const savedRef = useRef(false);
  useEffect(() => {
    if (savedRef.current) return;
    if (!results || !predictedClass) return;
    savedRef.current = true;

    const saveRecord = async () => {
      try {
        const {
          data: { user },
        } = await supabase.auth.getUser();
        if (!user) return;

        // Upload image to Supabase Storage with client-side encryption
        // Step 1: Optimize to WebP format (~40% size reduction)
        // Step 2: Encrypt with a device-bound key kept on the client
        // Store the path (not a public URL) so signed URLs can be generated at display time
        let image_url: string | null = null;
        let gradcam_image_url: string | null = null;

        if (image) {
          // Convert to WebP with 75% quality (1024px max dimension)
          const optimizedBlob = await optimizeImage(image, {
            maxDimension: 1024,
            quality: 0.75,
            format: 'image/webp',
          });

          // Encrypt the optimized image for privacy
          const encryptedBlob = await encryptImage(optimizedBlob, user.id);

          const ext = getEncryptedExtension();
          const path = `${user.id}/${caseId}.${ext}`;
          const { error: uploadErr } = await supabase.storage
            .from('analysis-images')
            .upload(path, encryptedBlob, { contentType: 'application/octet-stream' });
          if (!uploadErr) {
            image_url = path; // store path; HistoryScreen generates signed URLs + decrypts on fetch
          }
        }

        // Upload Grad-CAM heatmap if available
        if (gradcamImage) {
          try {
            // Optimize Grad-CAM image (same settings as original)
            const optimizedGradcamBlob = await optimizeImage(gradcamImage, {
              maxDimension: 1024,
              quality: 0.75,
              format: 'image/webp',
            });

            // Encrypt the Grad-CAM image
            const encryptedGradcamBlob = await encryptImage(optimizedGradcamBlob, user.id);

            const ext = getEncryptedExtension();
            const gradcamPath = `${user.id}/${caseId}_gradcam.${ext}`;
            const { error: gradcamUploadErr } = await supabase.storage
              .from('analysis-images')
              .upload(gradcamPath, encryptedGradcamBlob, {
                contentType: 'application/octet-stream',
              });

            if (!gradcamUploadErr) {
              gradcam_image_url = gradcamPath;
            }
          } catch (err) {
            // Non-blocking — if Grad-CAM upload fails, still save the analysis
            console.error('Failed to upload Grad-CAM image:', err);
          }
        }

        // Insert analysis record — id is the client-generated UUID so the
        // displayed Case ID always matches what is stored in the database.
        // Only store metadata fields that were actually provided (null for the rest).
        const metadataPayload = {
          age_approx: metadata?.ageApprox ?? null,
          sex: metadata?.sex ?? null,
          anatom_site: metadata?.anatomSite ?? null,
        };
        const hasAnyMetadata =
          metadataPayload.age_approx !== null ||
          metadataPayload.sex !== null ||
          metadataPayload.anatom_site !== null;

        const { error: insertErr } = await supabase.from('analyses').insert({
          id: caseId,
          user_id: user.id,
          image_url,
          gradcam_image_url,
          predicted_class_id: predictedClass.id,
          predicted_class_name: predictedClass.name,
          confidence: predictedClass.score,
          all_scores: results,
          metadata: hasAnyMetadata ? metadataPayload : null,
        });
        if (insertErr) throw insertErr;
        setSaveCompleted(true);
        // Invalidate cache so dashboard and history will fetch fresh data
        invalidateAll();
      } catch {
        // Non-blocking — show a banner so the user knows the record was not saved.
        setSaveFailed(true);
      }
    };

    saveRecord();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const saveNote = async () => {
    if (!saveCompleted) return;
    setSavingNote(true);
    setNoteError(null);
    try {
      const {
        data: { user },
      } = await supabase.auth.getUser();
      if (!user) throw new Error('Not authenticated');
      const trimmed = noteText.trim() || null;
      const { data: updated, error } = await supabase
        .from('analyses')
        .update({ notes: trimmed })
        .eq('id', caseId)
        .eq('user_id', user.id)
        .select('id');
      if (error) throw error;
      if (!updated || updated.length === 0) {
        // No rows updated - record may have been deleted or doesn't belong to user
        setNoteError('This analysis record no longer exists. It may have been deleted.');
        return;
      }
      setNoteSaved(true);
      setTimeout(() => setNoteSaved(false), 3000);
    } catch (err: unknown) {
      // Differentiate between database errors and other errors
      const errorMessage =
        err instanceof Error && err.message === 'Not authenticated'
          ? 'Authentication expired. Please log in again.'
          : 'Could not save note. Please check your connection and try again.';
      setNoteError(errorMessage);
    } finally {
      setSavingNote(false);
    }
  };

  const exportPdf = async () => {
    if (!predictedClass || !info) return;
    setExportingPdf(true);
    try {
      await generateDermatologyReport({
        caseId: caseIdDisplay,
        date: analysisDate,
        clinicianName: clinicianName || 'Unknown',
        classId: predictedClass.id,
        className: predictedClass.name,
        confidence: predictedClass.score,
        classInfo: info,
        allScores: classes,
        notes: noteText,
        imageDataUrl: image || undefined,
        metadata: metadata ?? null,
      });
    } finally {
      setExportingPdf(false);
    }
  };

  /* Skeleton loading state (Doherty Threshold) */
  if (loading) {
    return (
      <div className="flex-1 flex flex-col bg-slate-50 text-slate-900">
        <main className="max-w-6xl mx-auto w-full px-4 sm:px-6 py-8 space-y-6">
          <div className="h-40 bg-white border border-slate-200 rounded-xl animate-pulse" />
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            <div className="lg:col-span-3 h-72 bg-white border border-slate-200 rounded-xl animate-pulse" />
            <div className="lg:col-span-2 space-y-6">
              <div className="h-52 bg-white border border-slate-200 rounded-xl animate-pulse" />
              <div className="h-48 bg-white border border-slate-200 rounded-xl animate-pulse" />
            </div>
          </div>
        </main>
      </div>
    );
  }

  // Guard: show a minimal error state if results are missing
  if (!predictedClass || !info) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="text-center">
          <p className="text-slate-500 text-sm">No results available.</p>
          <button onClick={onAnalyzeAnother} className="mt-4 text-teal-600 font-medium text-sm">
            Try again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col bg-slate-50 text-slate-900">
      {/* MAIN CONTENT */}
      <main className="max-w-6xl mx-auto w-full px-4 sm:px-6 py-8 flex flex-col gap-6">
        {saveFailed && (
          <div
            role="alert"
            className="p-3 bg-amber-50 border border-amber-200 text-amber-700 text-xs rounded-lg font-medium"
          >
            This result could not be saved to your history. Please take a screenshot or note the
            result for your records.
          </div>
        )}
        {/* Case metadata bar */}
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-xs">
          <h1 className="text-sm font-semibold text-slate-500 uppercase tracking-widest">
            Analysis Result
          </h1>
          <div className="flex items-center gap-1.5">
            <span className="font-bold text-slate-400 uppercase tracking-widest">Case ID:</span>
            <span className="font-semibold text-slate-700 tabular-nums">{caseIdDisplay}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="font-bold text-slate-400 uppercase tracking-widest">Date:</span>
            <span className="font-semibold text-slate-700">{analysisDate}</span>
          </div>
        </div>

        {/* PATIENT METADATA — only rendered when at least one field is non-null */}
        {(() => {
          const chips: { label: string; value: string }[] = [];
          if (metadata?.ageApprox != null)
            chips.push({ label: 'Age', value: `${metadata.ageApprox} yrs` });
          if (metadata?.sex)
            chips.push({ label: 'Sex', value: metadata.sex.charAt(0).toUpperCase() + metadata.sex.slice(1) });
          if (metadata?.anatomSite)
            chips.push({
              label: 'Site',
              value: metadata.anatomSite
                .split('_')
                .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
                .join(' '),
            });
          if (chips.length === 0) return null;
          return (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mr-1">Patient</span>
              {chips.map((chip) => (
                <span
                  key={chip.label}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-teal-50 border border-teal-200 text-xs"
                >
                  <span className="font-semibold text-teal-500 uppercase tracking-wide text-[10px]">{chip.label}</span>
                  <span className="text-slate-700 font-medium">{chip.value}</span>
                </span>
              ))}
            </div>
          );
        })()}

        {/* PRIMARY RESULT CARD — includes analyzed image */}
        <ResultCard
          classId={predictedClass.id}
          className={predictedClass.name}
          confidence={predictedClass.score}
          info={info}
          imageUrl={image}
          gradcamImage={gradcamImage}
        />

        {/* PROBABILITY CHART — full width */}
        <ProbabilityChart classes={classes} predictedClassId={predictedClass.id} />

        {/* MEDICAL INFO — Full-width row below the bento pair */}
        <MedicalInfoCard info={info} />

        {/* CLINICIAN NOTES */}
        {!saveFailed && (
          <section className="bg-white rounded-xl border border-slate-400 shadow-sm p-5">
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">
              Clinician Notes
            </p>
            <textarea
              value={noteText}
              onChange={(e) => {
                setNoteText(e.target.value);
                setNoteSaved(false);
                setNoteError(null);
              }}
              rows={3}
              maxLength={2000}
              placeholder="Add clinical observations, follow-up plans, or patient notes…"
              className="w-full text-sm text-slate-700 border border-slate-300 rounded-lg p-3 resize-none focus:outline-none focus:ring-2 focus:ring-teal-400 focus:border-transparent placeholder:text-slate-300"
            />
            {noteError && <p className="text-xs text-red-500 font-medium mt-1">{noteError}</p>}
            <div className="flex items-center justify-between mt-2">
              <span className="text-xs text-slate-300">{noteText.length}/2000</span>
              <div className="flex items-center gap-2">
                {noteSaved && (
                  <span className="text-xs text-teal-600 font-medium">Note saved.</span>
                )}
                <button
                  onClick={saveNote}
                  disabled={savingNote || !noteText.trim() || !saveCompleted}
                  title={!saveCompleted ? 'Waiting for record to save…' : undefined}
                  className="text-xs font-semibold px-3 py-1.5 rounded-lg bg-teal-600 text-white hover:bg-teal-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-1.5"
                >
                  {savingNote ? (
                    <>
                      <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
                        <circle
                          className="opacity-25"
                          cx="12"
                          cy="12"
                          r="10"
                          stroke="currentColor"
                          strokeWidth="4"
                        />
                        <path
                          className="opacity-75"
                          fill="currentColor"
                          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                        />
                      </svg>
                      Saving…
                    </>
                  ) : !saveCompleted ? (
                    'Saving record…'
                  ) : (
                    'Save note'
                  )}
                </button>
              </div>
            </div>
          </section>
        )}

        {/* Action buttons — full width */}
        <div className="flex flex-col gap-3">
          <button
            onClick={exportPdf}
            disabled={exportingPdf}
            className="w-full py-2.5 px-4 rounded-full font-medium text-sm border-2 border-teal-600 text-teal-700 bg-teal-50 transition-all duration-200 hover:bg-teal-100 hover:border-teal-700 active:bg-teal-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-400 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span className="flex items-center justify-center gap-2">
              {exportingPdf ? (
                <>
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                    />
                  </svg>
                  Generating Report…
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  Download PDF Report
                </>
              )}
            </span>
          </button>
          <Button onClick={onAnalyzeAnother}>
            <span className="flex items-center justify-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              Analyze Another Image
            </span>
          </Button>
          <button
            onClick={onNavigateToHistory}
            className="w-full py-2.5 px-4 rounded-full font-medium text-sm border border-slate-200 text-slate-600 transition-all duration-200 hover:bg-teal-50 hover:border-teal-300 hover:text-teal-700 active:bg-teal-100 active:border-teal-400 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-400"
          >
            <span className="flex items-center justify-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              View Past Analyses
            </span>
          </button>
        </div>
      </main>

      {/* FOOTER — Clinical disclaimer (Tesler's Law: required but visually subordinate) */}
      <footer className="mt-auto py-6 border-t border-slate-100 bg-white">
        <div className="max-w-6xl mx-auto px-6">
          <p className="text-[11px] text-slate-400 leading-relaxed text-center">
            This tool generates probabilistic outputs from a machine learning model. It is designed
            to assist clinical decision-making and does not replace professional medical diagnosis.
            Always correlate with clinical findings and patient history.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default ResultsScreen;
