
import React, { useState, useEffect } from 'react';
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
  const [loadingMore,   setLoadingMore]   = useState(false);
  const [historyError,  setHistoryError]  = useState<string | null>(null);
  const [hasMore,       setHasMore]       = useState(false);
  const [page,          setPage]          = useState(0);

  const mapRows = async (data: Record<string, unknown>[]): Promise<AnalysisHistoryItem[]> => {
    // Generate signed URLs in one batched request for rows that have an image path stored
    const paths = data
      .map((row) => row.image_url as string | null)
      .filter((p): p is string => !!p && !p.startsWith('http'));

    let signedUrlMap: Record<string, string> = {};
    if (paths.length > 0) {
      const { data: signed } = await supabase.storage
        .from('analysis-images')
        .createSignedUrls(paths, 60 * 60 * 24); // 24-hour TTL
      if (signed) {
        for (const entry of signed) {
          if (entry.signedUrl) signedUrlMap[entry.path] = entry.signedUrl;
        }
      }
    }

    return data.map((row) => {
      const rawUrl = row.image_url as string | null;
      // Support legacy rows that stored full URLs directly
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
        allScores:  (row.all_scores as AnalysisHistoryItem['allScores']) ?? undefined,
      };
    });
  };

  // Initial load
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const { data, error } = await supabase
          .from('analyses')
          .select('id, created_at, predicted_class_id, predicted_class_name, confidence, image_url, all_scores')
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
    setLoadingMore(true);
    const nextPage = page + 1;
    try {
      const { data, error } = await supabase
        .from('analyses')
        .select('id, created_at, predicted_class_id, predicted_class_name, confidence, image_url, all_scores')
        .order('created_at', { ascending: false })
        .range(nextPage * PAGE_SIZE, (nextPage + 1) * PAGE_SIZE - 1);

      if (error) throw error;
      const rows = data ?? [];
      setHistoryItems((prev) => [...prev, ...(await mapRows(rows))]);
      setHasMore(rows.length === PAGE_SIZE);
      setPage(nextPage);
    } catch {
      setHistoryError('Could not load more records. Please try again.');
    } finally {
      setLoadingMore(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-slate-50 text-slate-900 pb-12">
      <main className="max-w-4xl mx-auto w-full px-4 sm:px-6 py-8 flex flex-col gap-8">
        <div className="flex items-center justify-between">
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
          <span className="text-[10px] font-bold text-slate-400 bg-slate-100 px-2.5 py-1 rounded uppercase tracking-wider">System Records</span>
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
          {/* Mobile card layout (< md) */}
          <ul className="divide-y divide-slate-100 md:hidden">
            {historyItems.map((item) => (
              <li key={item.id} className="flex items-center gap-4 px-4 py-4">
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
                <button
                  onClick={() => onViewDetails(item)}
                  className="shrink-0 text-teal-600 hover:text-teal-700 p-2 rounded-full hover:bg-teal-50 transition-colors"
                  title="View Details"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </li>
            ))}
          </ul>

          {/* Desktop table layout (md+) */}
          <div className="hidden md:block overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200">
                  <th className="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest">Lesion Context</th>
                  <th className="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest">Date / Time</th>
                  <th className="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest">Predicted Class</th>
                  <th className="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest text-right">Details</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {historyItems.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-50/50 transition-colors group">
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
                      <button 
                        onClick={() => onViewDetails(item)}
                        className="text-teal-600 hover:text-teal-700 text-sm font-bold uppercase tracking-widest flex items-center gap-1 justify-end ml-auto group-hover:translate-x-1 transition-transform"
                      >
                        View Details
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </button>
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
          Clinical Support Tool. System architecture v2.0.4-pro
        </p>
      </footer>
    </div>
  );
};

export default HistoryScreen;
