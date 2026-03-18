
import React, { useState, useEffect, useRef } from 'react';
import Button from './ui/Button';
import { supabase } from '../lib/supabase';
import type { AnalysisHistoryItem } from '../lib/types';

// Re-export so HistoryDetailScreen can still import from here
export type { AnalysisHistoryItem };

interface HistoryScreenProps {
  onBack: () => void;
  onViewDetails: (item: AnalysisHistoryItem) => void;
}

const PAGE_SIZE = 20;

const HistoryScreen: React.FC<HistoryScreenProps> = ({ onBack, onViewDetails }) => {
  const [historyItems, setHistoryItems] = useState<AnalysisHistoryItem[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [loadingMore,    setLoadingMore]    = useState(false);
  const [historyError,   setHistoryError]   = useState<string | null>(null);
  const [hasMore,        setHasMore]        = useState(false);
  const [page,           setPage]           = useState(0);
  const [userId,         setUserId]         = useState<string | null>(null);

  // Selection / deletion state
  const [selectMode,   setSelectMode]   = useState(false);
  const [selected,     setSelected]     = useState<Set<string>>(new Set());
  const [confirmModal, setConfirmModal] = useState<'selected' | 'all' | null>(null);
  const [deleting,     setDeleting]     = useState(false);

  const modalRef     = useRef<HTMLDivElement>(null);
  const cancelBtnRef = useRef<HTMLButtonElement>(null);

  const mapRows = async (data: Record<string, unknown>[]): Promise<AnalysisHistoryItem[]> => {
    const paths = data
      .map((row) => row.image_url as string | null)
      .filter((p): p is string => !!p && !p.startsWith('http'));

    let signedUrlMap: Record<string, string> = {};
    if (paths.length > 0) {
      const { data: signed } = await supabase.storage
        .from('analysis-images')
        .createSignedUrls(paths, 60 * 60); // 1-hour expiry
      if (signed) {
        for (const entry of signed) {
          if (entry.signedUrl) signedUrlMap[entry.path] = entry.signedUrl;
        }
      }
    }

    return data.map((row) => {
      const rawUrl = row.image_url as string | null;
      const imageUrl = rawUrl
        ? (rawUrl.startsWith('http') ? rawUrl : (signedUrlMap[rawUrl] ?? undefined))
        : undefined;
      return {
        id:         row.id as string,
        date:       new Date(row.created_at as string).toLocaleDateString('en-US', {
                      year: 'numeric', month: 'short', day: 'numeric',
                    }),
        time:       new Date(row.created_at as string).toLocaleTimeString('en-US', {
                      hour: '2-digit', minute: '2-digit',
                    }),
        classId:    row.predicted_class_id as string,
        className:  row.predicted_class_name as string,
        confidence: row.confidence as number,
        imageUrl,
        imagePath: (rawUrl && !rawUrl.startsWith('http')) ? rawUrl : undefined,
        allScores:  (row.all_scores as AnalysisHistoryItem['allScores']) ?? undefined,
        notes:      (row.notes as string | null) ?? undefined,
      };
    });
  };

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        const uid = user?.id ?? null;
        if (uid) setUserId(uid);

        const { data, error } = await supabase
          .from('analyses')
          .select('id, created_at, predicted_class_id, predicted_class_name, confidence, image_url, all_scores, notes')
          .eq('user_id', uid ?? '')
          .order('created_at', { ascending: false })
          .range(0, PAGE_SIZE - 1);

        if (error) throw error;
        const rows = data ?? [];
        setHistoryItems(await mapRows(rows));
        setHasMore(rows.length === PAGE_SIZE);
      } catch {
        setHistoryError('Could not load analysis history. Please try again.');
      } finally {
        setLoadingHistory(false);
      }
    };
    fetchHistory();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleLoadMore = async () => {
    if (!userId) return;
    setLoadingMore(true);
    const nextPage = page + 1;
    try {
      const { data, error } = await supabase
        .from('analyses')
        .select('id, created_at, predicted_class_id, predicted_class_name, confidence, image_url, all_scores, notes')
        .eq('user_id', userId)
        .order('created_at', { ascending: false })
        .range(nextPage * PAGE_SIZE, (nextPage + 1) * PAGE_SIZE - 1);

      if (error) throw error;
      const rows = data ?? [];
      const mappedRows = await mapRows(rows);
      setHistoryItems((prev) => [...prev, ...mappedRows]);
      setHasMore(rows.length === PAGE_SIZE);
      setPage(nextPage);
    } catch {
      setHistoryError('Could not load more records. Please try again.');
    } finally {
      setLoadingMore(false);
    }
  };

  // ── Selection helpers ──────────────────────────────────────────────────────
  const toggleSelectMode = () => {
    setSelectMode(prev => !prev);
    setSelected(new Set());
  };

  const toggleSelect = (id: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const allSelected = historyItems.length > 0 && historyItems.every(i => selected.has(i.id));

  const toggleSelectAll = () => {
    setSelected(allSelected ? new Set() : new Set(historyItems.map(i => i.id)));
  };

  // ── Delete handlers ────────────────────────────────────────────────────────
  const handleDeleteSelected = async () => {
    setDeleting(true);
    setHistoryError(null);
    const ids = [...selected];

    try {
      const paths = ids
        .map(id => historyItems.find(i => i.id === id)?.imagePath)
        .filter((p): p is string => !!p);

      if (paths.length > 0) {
        const { error: storageError } = await supabase.storage.from('analysis-images').remove(paths);
        if (storageError) {
          throw new Error(`Failed to remove images: ${storageError.message}`);
        }
      }

      const { error: dbError } = await supabase.from('analyses').delete().in('id', ids);
      if (dbError) {
        throw new Error(`Failed to delete records: ${dbError.message}`);
      }

      setHistoryItems(prev => prev.filter(i => !ids.includes(i.id)));
      setSelected(new Set());
      setSelectMode(false);
      setConfirmModal(null);
    } catch (err) {
      setHistoryError(
        err instanceof Error ? err.message : 'Could not delete selected records. Please try again.',
      );
    } finally {
      setDeleting(false);
    }
  };

  const handleClearAll = async () => {
    setDeleting(true);
    setHistoryError(null);

    try {
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      if (userError || !user) throw new Error('Not authenticated');

      const { data: allRows, error: fetchError } = await supabase
        .from('analyses')
        .select('image_url')
        .eq('user_id', user.id)
        .not('image_url', 'is', null);
      if (fetchError) {
        throw new Error(`Failed to fetch records: ${fetchError.message}`);
      }

      const paths = (allRows ?? [])
        .map(r => r.image_url as string)
        .filter(p => !p.startsWith('http'));
      if (paths.length > 0) {
        const { error: storageError } = await supabase.storage.from('analysis-images').remove(paths);
        if (storageError) {
          throw new Error(`Failed to remove images: ${storageError.message}`);
        }
      }

      const { error: dbError } = await supabase.from('analyses').delete().eq('user_id', user.id);
      if (dbError) {
        throw new Error(`Failed to clear history: ${dbError.message}`);
      }

      setHistoryItems([]);
      setSelected(new Set());
      setSelectMode(false);
      setHasMore(false);
      setConfirmModal(null);
    } catch (err) {
      setHistoryError(
        err instanceof Error ? err.message : 'Could not clear history. Please try again.',
      );
    } finally {
      setDeleting(false);
    }
  };

  const hasItems = historyItems.length > 0;

  // Modal: Escape-to-close, focus trapping, and initial focus
  useEffect(() => {
    if (!confirmModal) return;

    cancelBtnRef.current?.focus();

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !deleting) {
        setConfirmModal(null);
        return;
      }
      if (e.key === 'Tab' && modalRef.current) {
        const focusable = modalRef.current.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        );
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last  = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [confirmModal, deleting]);

  return (
    <div className="flex-1 flex flex-col bg-slate-50 text-slate-900 pb-12">
      <main className="max-w-4xl mx-auto w-full px-4 sm:px-6 py-8 flex flex-col gap-8">

        {/* Header */}
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <button
              onClick={onBack}
              className="p-2 hover:bg-slate-100 rounded-full transition-colors text-slate-400 hover:text-slate-600"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Past Analyses</h1>
          </div>

          {/* Action buttons — only show when records exist */}
          {!loadingHistory && hasItems && (
            <div className="flex items-center gap-2">
              <button
                onClick={toggleSelectMode}
                className={[
                  'text-xs font-bold px-3 py-1.5 rounded-lg border transition-colors',
                  selectMode
                    ? 'bg-slate-800 text-white border-slate-800'
                    : 'bg-white text-slate-600 border-slate-200 hover:border-slate-300 hover:text-slate-800',
                ].join(' ')}
              >
                {selectMode ? 'Cancel' : 'Select'}
              </button>
              {!selectMode && (
                <button
                  onClick={() => setConfirmModal('all')}
                  className="text-xs font-bold px-3 py-1.5 rounded-lg border border-red-200 text-red-500 bg-white hover:bg-red-50 hover:border-red-300 transition-colors"
                >
                  Clear History
                </button>
              )}
            </div>
          )}
        </div>

        <section className="bg-white rounded-3xl border border-slate-200 overflow-hidden shadow-sm">
          {loadingHistory ? (
            <div className="flex items-center justify-center py-16">
              <svg className="animate-spin h-6 w-6 text-teal-600" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            </div>
          ) : historyError ? (
            <div className="py-12 px-6 text-center">
              <p className="text-sm text-red-500 font-medium">{historyError}</p>
            </div>
          ) : historyItems.length === 0 ? (
            <div className="py-16 px-6 text-center">
              <p className="text-sm text-slate-400">No analyses found.</p>
              <p className="text-xs text-slate-300 mt-1">Run your first classification to see it here.</p>
            </div>
          ) : (<>

          {/* Selection toolbar */}
          {selectMode && (
            <div className="flex items-center justify-between px-4 sm:px-6 py-3 border-b border-slate-200 bg-slate-50">
              <label className="flex items-center gap-2.5 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={allSelected}
                  onChange={toggleSelectAll}
                  className="w-4 h-4 rounded border-slate-300 text-teal-600 accent-teal-600 cursor-pointer"
                />
                <span className="text-xs font-semibold text-slate-600">
                  {allSelected ? 'Deselect all' : 'Select all'}
                </span>
              </label>
              <button
                disabled={selected.size === 0}
                onClick={() => setConfirmModal('selected')}
                className="text-xs font-bold px-3 py-1.5 rounded-lg bg-red-500 text-white hover:bg-red-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                Delete {selected.size > 0 ? `${selected.size} selected` : 'selected'}
              </button>
            </div>
          )}

          {/* Mobile card layout (< md) */}
          <ul className="divide-y divide-slate-100 md:hidden">
            {historyItems.map((item) => (
              <li
                key={item.id}
                className={['flex items-center gap-4 px-4 py-4 transition-colors', selectMode && selected.has(item.id) ? 'bg-red-50/60' : ''].join(' ')}
              >
                {selectMode && (
                  <input
                    type="checkbox"
                    checked={selected.has(item.id)}
                    onChange={() => toggleSelect(item.id)}
                    aria-label={`Select case DRM-${item.id.slice(0, 8).toUpperCase()}`}
                    className="w-4 h-4 rounded border-slate-300 accent-teal-600 cursor-pointer shrink-0"
                  />
                )}
                <div className="w-12 h-12 rounded-lg bg-slate-100 flex items-center justify-center text-slate-300 border border-slate-200 overflow-hidden shrink-0">
                  {item.imageUrl ? (
                    <img src={item.imageUrl} alt="Lesion thumbnail" className="w-full h-full object-cover" />
                  ) : (
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-sm font-bold text-teal-700 uppercase">{item.classId}</span>
                    <span className="text-[10px] font-bold text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded tracking-tighter">{item.confidence.toFixed(1)}%</span>
                  </div>
                  <p className="text-xs text-slate-500 truncate">{item.className}</p>
                  <p className="text-xs text-slate-400 mt-0.5">{item.date} · {item.time}</p>
                </div>
                {!selectMode && (
                  <button
                    onClick={() => onViewDetails(item)}
                    className="shrink-0 text-teal-600 hover:text-teal-700 p-2 rounded-full hover:bg-teal-50 transition-colors"
                    title="View Details"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </button>
                )}
              </li>
            ))}
          </ul>

          {/* Desktop table layout (md+) */}
          <div className="hidden md:block overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200">
                  {selectMode && <th className="pl-6 py-4 w-10" />}
                  <th className="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest">Lesion Context</th>
                  <th className="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest">Date / Time</th>
                  <th className="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest">Predicted Class</th>
                  <th className="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest text-right">Details</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {historyItems.map((item) => (
                  <tr
                    key={item.id}
                    className={['hover:bg-slate-50/50 transition-colors group', selectMode && selected.has(item.id) ? 'bg-red-50/60' : ''].join(' ')}
                  >
                    {selectMode && (
                      <td className="pl-6 py-4 w-10">
                        <input
                          type="checkbox"
                          checked={selected.has(item.id)}
                          onChange={() => toggleSelect(item.id)}
                          aria-label={`Select case DRM-${item.id.slice(0, 8).toUpperCase()}`}
                          className="w-4 h-4 rounded border-slate-300 accent-teal-600 cursor-pointer"
                        />
                      </td>
                    )}
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-lg bg-slate-100 flex items-center justify-center text-slate-300 border border-slate-200 overflow-hidden relative">
                          {item.imageUrl ? (
                            <img src={item.imageUrl} alt="Lesion thumbnail" className="w-full h-full object-cover" />
                          ) : (
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                          )}
                        </div>
                        <div>
                          <div className="text-xs font-bold text-slate-400 uppercase tracking-tighter">Case Reference</div>
                          <div className="text-sm font-semibold text-slate-700 font-mono">DRM-{item.id.slice(0, 8).toUpperCase()}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex flex-col">
                        <span className="text-sm font-medium text-slate-700">{item.date}</span>
                        <span className="text-xs text-slate-400">{item.time}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex flex-col">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-bold text-teal-700 uppercase">{item.classId}</span>
                          <span className="text-[10px] font-bold text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded tracking-tighter">{item.confidence.toFixed(1)}%</span>
                        </div>
                        <span className="text-xs text-slate-500">{item.className}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-right">
                      {!selectMode && (
                        <button
                          onClick={() => onViewDetails(item)}
                          className="text-teal-600 hover:text-teal-700 text-sm font-bold uppercase tracking-widest flex items-center gap-1 justify-end ml-auto group-hover:translate-x-1 transition-transform"
                        >
                          View Details
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Load More */}
          {hasMore && (
            <div className="px-6 py-5 border-t border-slate-100 flex justify-center">
              <button
                onClick={handleLoadMore}
                disabled={loadingMore}
                className="flex items-center gap-2 text-sm font-semibold text-teal-600 hover:text-teal-700 disabled:opacity-50 transition-colors"
              >
                {loadingMore ? (
                  <>
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Loading…
                  </>
                ) : (
                  <>
                    Load more records
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </>
                )}
              </button>
            </div>
          )}
          </>)}
        </section>

        <div className="p-6 bg-blue-50/30 rounded-2xl border border-blue-100">
          <div className="flex gap-3">
            <div className="text-blue-500">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <p className="text-xs text-blue-700 font-medium leading-relaxed">
              Analysis history is fetched from your account, ordered newest first. Records are persisted across devices and sessions.
            </p>
          </div>
        </div>

        <div className="flex justify-center">
          <div className="max-w-xs w-full">
            <Button variant="secondary" onClick={onBack}>
              Back to Dashboard
            </Button>
          </div>
        </div>
      </main>

      <footer className="mt-auto py-10 text-center border-t border-slate-100 bg-white">
        <p className="text-[11px] font-medium text-slate-500 uppercase tracking-widest leading-relaxed">
          Clinical Support Tool. Designed to assist medical professionals.
        </p>
      </footer>

      {/* ── Confirmation modal ─────────────────────────────────────────────── */}
      {confirmModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/40 backdrop-blur-sm"
            onClick={() => !deleting && setConfirmModal(null)}
          />
          {/* Dialog */}
          <div
            ref={modalRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-confirm-title"
            aria-describedby="delete-confirm-desc"
            className="relative bg-white rounded-2xl shadow-xl max-w-sm w-full p-6 flex flex-col gap-5"
          >
            {/* Icon */}
            <div className="flex items-center justify-center w-12 h-12 rounded-full bg-red-50 mx-auto">
              <svg className="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </div>
            {/* Text */}
            <div className="text-center">
              <h2 id="delete-confirm-title" className="text-base font-bold text-slate-900 mb-1">
                {confirmModal === 'all' ? 'Clear all history?' : `Delete ${selected.size} record${selected.size !== 1 ? 's' : ''}?`}
              </h2>
              <p id="delete-confirm-desc" className="text-sm text-slate-500 leading-relaxed">
                {confirmModal === 'all'
                  ? 'This will permanently delete all your analysis records. This action cannot be undone.'
                  : `This will permanently delete the selected record${selected.size !== 1 ? 's' : ''}. This action cannot be undone.`}
              </p>
            </div>
            {/* Actions */}
            <div className="flex gap-3">
              <button
                ref={cancelBtnRef}
                onClick={() => setConfirmModal(null)}
                disabled={deleting}
                className="flex-1 py-2.5 rounded-xl border border-slate-200 text-sm font-semibold text-slate-600 hover:bg-slate-50 disabled:opacity-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmModal === 'all' ? handleClearAll : handleDeleteSelected}
                disabled={deleting}
                className="flex-1 py-2.5 rounded-xl bg-red-500 text-sm font-bold text-white hover:bg-red-600 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
              >
                {deleting ? (
                  <>
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Deleting…
                  </>
                ) : 'Delete permanently'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HistoryScreen;
