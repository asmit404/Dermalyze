import React, { useState, useEffect, useRef } from 'react';
import ResultCard from './ui/ResultCard';
import ProbabilityChart from './ui/ProbabilityChart';
import MedicalInfoCard from './ui/MedicalInfoCard';
import { classInfoMap } from '../lib/classInfo';
import { supabase } from '../lib/supabase';
import { generateDermatologyReport } from '../lib/pdfReportGenerator';
import type { AnalysisHistoryItem } from '../lib/types';

interface HistoryDetailScreenProps {
  item: AnalysisHistoryItem | null;
  onBack: () => void;
}

/**
 * HistoryDetailScreen — Historical analysis record viewer.
 *
 * Shares identical visual structure with ResultsScreen (Jakob's Law —
 * consistency within the product). Reuses the same sub-components:
 * ResultCard, ProbabilityChart, ImageCard, MedicalInfoCard.
 *
 * Additional UX considerations:
 * - Back navigation is prominent in the header (Fitts's Law).
 * - Case ID and timestamp are surfaced in the header for clinical
 *   record-keeping (Jakob's Law — matches EHR conventions).
 * - Skeleton loading state for perceived speed (Doherty Threshold).
 */
const HistoryDetailScreen: React.FC<HistoryDetailScreenProps> = ({ item, onBack }) => {
  const [loading, setLoading] = useState(true);

  // Note state
  const [noteText,    setNoteText]    = useState(item?.notes ?? '');
  const [editing,     setEditing]     = useState(false);
  const [editText,    setEditText]    = useState('');
  const [savingNote,  setSavingNote]  = useState(false);
  const [noteSaved,   setNoteSaved]   = useState(false);
  const [noteError,   setNoteError]   = useState<string | null>(null);
  const [clinicianName, setClinicianName] = useState<string>('');
  const [exportingPdf,  setExportingPdf]  = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 300);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!item) onBack();
  }, [item, onBack]);

  // Sync noteText if the item's note changes externally (e.g. after a fresh fetch)
  useEffect(() => {
    if (!editing) setNoteText(item?.notes ?? '');
  }, [item?.id, item?.notes]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch clinician name for PDF report
  useEffect(() => {
    const fetchClinicianName = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        const name = user.user_metadata?.full_name || user.user_metadata?.name || user.email || 'Unknown';
        setClinicianName(name);
      }
    };
    fetchClinicianName();
  }, []);

  if (!item) return null;

  const startEditing = () => {
    setEditText(noteText);
    setEditing(true);
    setNoteSaved(false);
    setNoteError(null);
    setTimeout(() => textareaRef.current?.focus(), 0);
  };

  const cancelEditing = () => {
    setEditing(false);
    setNoteError(null);
  };

  const saveNote = async () => {
    setSavingNote(true);
    setNoteError(null);
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error('Not authenticated');
      const trimmed = editText.trim() || null;
      const { data: updated, error } = await supabase
        .from('analyses')
        .update({ notes: trimmed })
        .eq('id', item.id)
        .eq('user_id', user.id)
        .select('id');
      if (error) throw error;
      if (!updated || updated.length === 0) {
        // No rows updated - record may have been deleted or doesn't belong to user
        setNoteError('This analysis record no longer exists. It may have been deleted.');
        return;
      }
      setNoteText(trimmed ?? '');
      setEditing(false);
      setNoteSaved(true);
      setTimeout(() => setNoteSaved(false), 3000);
    } catch (err: unknown) {
      // Differentiate between database errors and other errors
      const errorMessage = err instanceof Error && err.message === 'Not authenticated'
        ? 'Authentication expired. Please log in again.'
        : 'Could not save note. Please check your connection and try again.';
      setNoteError(errorMessage);
    } finally {
      setSavingNote(false);
    }
  };

  const exportPdf = async () => {
    if (!item || !info) return;
    setExportingPdf(true);
    try {
      await generateDermatologyReport({
        caseId,
        date: item.date,
        time: item.time,
        clinicianName: clinicianName || 'Unknown',
        classId: item.classId,
        className: item.className,
        confidence: item.confidence,
        classInfo: info,
        allScores: classes,
        notes: noteText || undefined,
        imageDataUrl: item.imageUrl || undefined,
      });
    } finally {
      setExportingPdf(false);
    }
  };

  // Only use real scores from the DB — never fabricate a distribution.
  const classes = item.allScores ?? null;

  const info = classInfoMap[item.classId];
  const caseId = `DRM-${item.id.slice(0, 8).toUpperCase()}`;

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

  return (
    <div className="flex-1 flex flex-col bg-slate-50 text-slate-900">
      {/* MAIN CONTENT */}
      <main className="max-w-6xl mx-auto w-full px-4 sm:px-6 py-8 flex flex-col gap-6">
        {/* Case metadata bar */}
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-xs">
          <h1 className="text-sm font-semibold text-slate-500 uppercase tracking-widest">Analysis Record</h1>
          <div className="flex items-center gap-1.5">
            <span className="font-bold text-slate-400 uppercase tracking-widest">Case ID:</span>
            <span className="font-semibold text-slate-700 tabular-nums">{caseId}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="font-bold text-slate-400 uppercase tracking-widest">Date:</span>
            <span className="font-semibold text-slate-700">{item.date} &middot; {item.time}</span>
          </div>
        </div>

        {/* PRIMARY RESULT CARD — includes analyzed image */}
        <ResultCard
          classId={item.classId}
          className={item.className}
          confidence={item.confidence}
          info={info}
          imageUrl={item.imageUrl}
        />

        {/* PROBABILITY CHART — only render when real scores are available */}
        {classes ? (
          <ProbabilityChart classes={classes} predictedClassId={item.classId} />
        ) : (
          <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Probability Breakdown</p>
            <p className="text-sm text-slate-500">Full probability breakdown is not available for this record.</p>
          </section>
        )}

        {/* MEDICAL INFO — Full-width row below the bento pair */}
        {info ? (
          <MedicalInfoCard info={info} />
        ) : (
          <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
            <p className="text-sm text-slate-500">
              No condition information available for this classification.
            </p>
          </section>
        )}

        {/* NOTES SECTION */}
        <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
          <div className="flex items-center justify-between mb-3">
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Clinician Notes</p>
            {!editing && (
              <button
                onClick={startEditing}
                className="text-xs font-semibold text-teal-600 hover:text-teal-700 flex items-center gap-1 px-2 py-1 rounded-lg hover:bg-teal-50 transition-colors"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536M9 13l6.586-6.586a2 2 0 012.828 2.828L11.828 15.828a2 2 0 01-1.414.586H9v-2.414a2 2 0 01.586-1.414z" />
                </svg>
                {noteText ? 'Edit' : 'Add note'}
              </button>
            )}
          </div>

          {editing ? (
            <div className="flex flex-col gap-3">
              <textarea
                ref={textareaRef}
                value={editText}
                onChange={(e) => setEditText(e.target.value)}
                rows={4}
                maxLength={2000}
                placeholder="Add clinical observations, follow-up plans, or patient notes…"
                className="w-full text-sm text-slate-700 border border-slate-200 rounded-lg p-3 resize-none focus:outline-none focus:ring-2 focus:ring-teal-400 focus:border-transparent placeholder:text-slate-300"
              />
              {noteError && (
                <p className="text-xs text-red-500 font-medium">{noteError}</p>
              )}
              <div className="flex items-center justify-between gap-3">
                <span className="text-xs text-slate-300">{editText.length}/2000</span>
                <div className="flex gap-2">
                  <button
                    onClick={cancelEditing}
                    disabled={savingNote}
                    className="text-xs font-semibold px-3 py-1.5 rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-50 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={saveNote}
                    disabled={savingNote}
                    className="text-xs font-semibold px-3 py-1.5 rounded-lg bg-teal-600 text-white hover:bg-teal-700 disabled:opacity-50 transition-colors flex items-center gap-1.5"
                  >
                    {savingNote ? (
                      <>
                        <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                        Saving…
                      </>
                    ) : 'Save'}
                  </button>
                </div>
              </div>
            </div>
          ) : noteText ? (
            <div className="flex flex-col gap-1.5">
              <p className="text-sm text-slate-700 whitespace-pre-wrap leading-relaxed">{noteText}</p>
              {noteSaved && (
                <p className="text-xs text-teal-600 font-medium">Note saved.</p>
              )}
            </div>
          ) : (
            <p className="text-sm text-slate-400 italic">No notes for this record.</p>
          )}
        </section>

        {/* Action buttons */}
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
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Generating Report…
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Download PDF Report
                </>
              )}
            </span>
          </button>
          <button
            onClick={onBack}
            className="w-full py-2.5 px-4 rounded-full font-medium text-sm border border-slate-200 text-slate-600 transition-all duration-200 hover:bg-teal-50 hover:border-teal-300 hover:text-teal-700 active:bg-teal-100 active:border-teal-400 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-400"
          >
            <span className="flex items-center justify-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Back to History
            </span>
          </button>
        </div>
      </main>

      {/* FOOTER — Clinical Disclaimer */}
      <footer className="mt-auto py-6 border-t border-slate-100 bg-white">
        <div className="max-w-6xl mx-auto px-6">
          <p className="text-[11px] text-slate-400 leading-relaxed text-center">
            Historical records are provided for longitudinal review. Classification scores are
            probabilistic outputs and serve as diagnostic aids only. Always correlate with clinical
            findings and professional judgment.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default HistoryDetailScreen;
