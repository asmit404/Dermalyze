import React, { useState, useEffect, useMemo, useRef } from 'react';
import Button from './ui/Button';
import ResultCard from './ui/ResultCard';
import ProbabilityChart from './ui/ProbabilityChart';
import MedicalInfoCard from './ui/MedicalInfoCard';
import { classInfoMap } from '../lib/classInfo';
import { supabase } from '../lib/supabase';
import type { ClassResult } from '../lib/types';

interface ResultsScreenProps {
  image: string | null;
  results: ClassResult[] | null;
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
  onAnalyzeAnother,
  onNavigateToHistory,
}) => {
  const [loading, setLoading] = useState(true);

  // Doherty Threshold: skeleton -> content in < 400ms
  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 350);
    return () => clearTimeout(timer);
  }, []);

  // Use real results from the API; empty array fallback keeps the UI safe
  const classes: ClassResult[] = results ?? [];

  const predictedClass = classes.length
    ? classes.reduce((prev, cur) => (prev.score > cur.score ? prev : cur))
    : null;

  const info = predictedClass ? classInfoMap[predictedClass.id] : undefined;

  // Stable UUID v4 generated once per mount — used as the actual DB primary key
  // and image storage path, so the displayed ID always matches the real record.
  const caseId = useMemo(() => crypto.randomUUID(), []);
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
        const { data: { user } } = await supabase.auth.getUser();
        if (!user) return;

        // Upload image to Supabase Storage
        // Store the path (not a public URL) so signed URLs can be generated at display time
        let image_url: string | null = null;
        if (image) {
          const blob = await (await fetch(image)).blob();
          const ext = blob.type === 'image/png' ? 'png' : 'jpg';
          const path = `${user.id}/${caseId}.${ext}`;
          const { error: uploadErr } = await supabase.storage
            .from('analysis-images')
            .upload(path, blob, { contentType: blob.type });
          if (!uploadErr) {
            image_url = path; // store path; HistoryScreen generates signed URLs on fetch
          }
        }

        // Insert analysis record — id is the client-generated UUID so the
        // displayed Case ID always matches what is stored in the database.
        await supabase.from('analyses').insert({
          id: caseId,
          user_id: user.id,
          image_url,
          predicted_class_id: predictedClass.id,
          predicted_class_name: predictedClass.name,
          confidence: predictedClass.score,
          all_scores: results,
        });
      } catch (err) {
        // Non-blocking — failed persistence should not disrupt the results view
        console.warn('Failed to save analysis record:', err);
      }
    };

    saveRecord();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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
        {/* Case metadata bar */}
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-xs">
          <h1 className="text-sm font-semibold text-slate-500 uppercase tracking-widest">Analysis Result</h1>
          <div className="flex items-center gap-1.5">
            <span className="font-bold text-slate-400 uppercase tracking-widest">Case ID:</span>
            <span className="font-semibold text-slate-700 tabular-nums">{caseIdDisplay}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="font-bold text-slate-400 uppercase tracking-widest">Date:</span>
            <span className="font-semibold text-slate-700">{analysisDate}</span>
          </div>
        </div>

        {/* PRIMARY RESULT CARD — includes analyzed image */}
        <ResultCard
          classId={predictedClass.id}
          className={predictedClass.name}
          confidence={predictedClass.score}
          info={info}
          imageUrl={image}
        />

        {/* PROBABILITY CHART — full width */}
        <ProbabilityChart classes={classes} predictedClassId={predictedClass.id} />

        {/* MEDICAL INFO — Full-width row below the bento pair */}
        <MedicalInfoCard info={info} />

        {/* Action buttons — full width */}
        <div className="flex flex-col gap-3">
          <Button onClick={onAnalyzeAnother}>
            <span className="flex items-center justify-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
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
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
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
