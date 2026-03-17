
import React, { useState, useEffect } from 'react';
import Button from './ui/Button';
import { supabase } from '../lib/supabase';

interface DashboardScreenProps {
  onNavigateToUpload: () => void;
  onNavigateToHistory: () => void;
}

// Classes that warrant clinical follow-up — kept for reference, filtering done server-side via RPC
const SHORT_CLASS_NAMES: Record<string, string> = {
  akiec: 'Actinic Keratoses',
  bcc:   'Basal Cell Carcinoma',
  bkl:   'Benign Keratosis',
  df:    'Dermatofibroma',
  mel:   'Melanoma',
  nv:    'Melanocytic Nevi',
  vasc:  'Vascular Lesions',
};

// Bar colour per class — risk-coded
const BAR_COLORS: Record<string, string> = {
  mel:   'bg-red-400',
  bcc:   'bg-orange-400',
  akiec: 'bg-amber-400',
  bkl:   'bg-teal-400',
  df:    'bg-teal-400',
  nv:    'bg-teal-400',
  vasc:  'bg-teal-400',
};

const RISK_LABEL: Record<string, { label: string; cls: string }> = {
  mel:   { label: 'Critical', cls: 'text-red-600 bg-red-50 border-red-200' },
  bcc:   { label: 'High',     cls: 'text-orange-600 bg-orange-50 border-orange-200' },
  akiec: { label: 'Moderate', cls: 'text-amber-600 bg-amber-50 border-amber-200' },
  bkl:   { label: 'Low',      cls: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
  df:    { label: 'Low',      cls: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
  nv:    { label: 'Low',      cls: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
  vasc:  { label: 'Low',      cls: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
};

interface RpcResult {
  total:          number;
  this_month:     number;
  avg_confidence: number | null;
  needs_review:   number;
  class_counts:   { id: string; name: string; count: number }[] | null;
}

interface DashStats {
  total: number;
  thisMonth: number;
  avgConfidence: number | null;
  needsReview: number;
  classCounts: { id: string; name: string; count: number }[];
  lastAnalysis: {
    className: string;
    classId: string;
    confidence: number;
    date: string;
    imageUrl: string | null;
  } | null;
}

const DashboardScreen: React.FC<DashboardScreenProps> = ({
  onNavigateToUpload,
  onNavigateToHistory,
}) => {
  const [stats,    setStats]    = useState<DashStats | null>(null);
  const [loading,  setLoading]  = useState(true);
  const [fetchErr, setFetchErr] = useState(false);
  const [userName, setUserName] = useState('');

  useEffect(() => {
    const load = async () => {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        if (user) setUserName(user.user_metadata?.full_name?.split(' ')[0] ?? '');

        const [
          { data: rpcData,  error: rpcErr },
          { data: lastRows, error: lastErr },
        ] = await Promise.all([
          supabase.rpc('get_dashboard_stats'),
          supabase
            .from('analyses')
            .select('predicted_class_id, predicted_class_name, confidence, created_at, image_url')
            .eq('user_id', user?.id ?? '')
            .order('created_at', { ascending: false })
            .limit(1),
        ]);

        if (rpcErr || !rpcData) { setFetchErr(true); setLoading(false); return; }

        const rpc  = rpcData as RpcResult;
        const last = (!lastErr && lastRows && lastRows.length > 0) ? lastRows[0] : null;

        let resolvedImageUrl: string | null = last?.image_url ?? null;
        if (resolvedImageUrl && !resolvedImageUrl.startsWith('http')) {
          const { data: signed } = await supabase.storage
            .from('analysis-images')
            .createSignedUrl(resolvedImageUrl, 60 * 60); // 1-hour expiry
          resolvedImageUrl = signed?.signedUrl ?? null;
        }

        const lastAnalysis = last ? {
          className:  SHORT_CLASS_NAMES[last.predicted_class_id] ?? last.predicted_class_name,
          classId:    last.predicted_class_id,
          confidence: last.confidence,
          date: new Date(last.created_at).toLocaleDateString('en-GB', {
            day: '2-digit', month: 'short', year: 'numeric',
          }),
          imageUrl: resolvedImageUrl,
        } : null;

        const classCounts = (rpc.class_counts ?? []).map(c => ({
          id:    c.id,
          name:  SHORT_CLASS_NAMES[c.id] ?? c.name,
          count: c.count,
        }));

        setStats({
          total:         rpc.total,
          thisMonth:     rpc.this_month,
          avgConfidence: rpc.avg_confidence,
          needsReview:   rpc.needs_review,
          classCounts,
          lastAnalysis,
        });
      } catch {
        setFetchErr(true);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const greeting = () => {
    const h = new Date().getHours();
    if (h < 12) return 'Good morning';
    if (h < 17) return 'Good afternoon';
    return 'Good evening';
  };

  // ── Skeleton ────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="flex-1 flex flex-col bg-slate-50">
        <main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 py-8 flex flex-col gap-6">
          <div className="h-8 w-48 bg-slate-200 rounded-lg animate-pulse" />
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-24 bg-white border border-slate-200 rounded-2xl animate-pulse" />
            ))}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            <div className="lg:col-span-3 h-64 bg-white border border-slate-200 rounded-2xl animate-pulse" />
            <div className="lg:col-span-2 h-64 bg-white border border-slate-200 rounded-2xl animate-pulse" />
          </div>
        </main>
      </div>
    );
  }

  const maxCount = stats ? Math.max(...stats.classCounts.map(c => c.count), 1) : 1;

  // ── Fetch error state ───────────────────────────────────────────────────
  if (fetchErr) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="text-center max-w-sm">
          <p className="text-sm font-medium text-slate-600 mb-1">Could not load dashboard data.</p>
          <p className="text-xs text-slate-400">Check your connection and refresh the page.</p>
        </div>
      </div>
    );
  }

  // ── Empty state (new user) ──────────────────────────────────────────────
  if (!stats || stats.total === 0) {
    return (
      <div className="flex-1 flex flex-col bg-slate-50">
        <main className="flex-1 flex items-center justify-center p-6 sm:p-12">
          <div className="max-w-lg w-full text-center">
            <div className="bg-white rounded-3xl border border-slate-200 p-10 sm:p-16 shadow-sm">
              <div className="mb-6 inline-flex items-center justify-center w-16 h-16 bg-teal-50 rounded-full text-teal-600">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <h2 className="text-2xl font-semibold text-slate-900 mb-3 tracking-wide">
                {greeting()}{userName ? `, ${userName}.` : ''}
              </h2>
              <p className="text-slate-500 text-base mb-8 max-w-sm mx-auto leading-relaxed">
                No analyses yet. Upload your first dermatoscopic image to begin.
              </p>
              <Button onClick={onNavigateToUpload}>
                <div className="flex items-center justify-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                  Upload First Image
                </div>
              </Button>
            </div>
          </div>
        </main>
      </div>
    );
  }

  // ── Full dashboard ──────────────────────────────────────────────────────
  return (
    <div className="flex-1 flex flex-col bg-slate-50">
      <main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 py-8 flex flex-col gap-6 pb-12">

        {/* Greeting */}
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            {greeting()}{userName ? `, ${userName}` : ''}
          </h1>
          <p className="text-sm text-slate-400 mt-0.5">Clinical analysis summary</p>
        </div>

        {/* ── Stat strip ── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Total */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Total Analyses</p>
            <p className="text-3xl font-bold text-slate-900 tracking-tight">{stats.total}</p>
          </div>

          {/* This month */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">This Month</p>
            <p className="text-3xl font-bold text-teal-600 tracking-tight">{stats.thisMonth}</p>
          </div>

          {/* Avg confidence */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Avg Confidence</p>
            <p className={[
              'text-3xl font-bold tracking-tight',
              stats.avgConfidence !== null && stats.avgConfidence >= 80 ? 'text-emerald-600'
                : stats.avgConfidence !== null && stats.avgConfidence >= 50 ? 'text-amber-600'
                : 'text-red-600',
            ].join(' ')}>
              {stats.avgConfidence !== null ? `${stats.avgConfidence}%` : '—'}
            </p>
          </div>

          {/* Needs review */}
          <div className={[
            'rounded-2xl border shadow-sm p-5',
            stats.needsReview > 0 ? 'bg-amber-50 border-amber-200' : 'bg-white border-slate-200',
          ].join(' ')}>
            <p className={[
              'text-[10px] font-bold uppercase tracking-widest mb-1',
              stats.needsReview > 0 ? 'text-amber-600' : 'text-slate-400',
            ].join(' ')}>Needs Review</p>
            <p className={[
              'text-3xl font-bold tracking-tight',
              stats.needsReview > 0 ? 'text-amber-700' : 'text-slate-900',
            ].join(' ')}>{stats.needsReview}</p>
            <p className="text-[10px] text-amber-500 mt-0.5 font-medium">
              {stats.needsReview > 0 ? 'MEL · BCC · AK' : 'No flags'}
            </p>
          </div>
        </div>

        {/* ── Main grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">

          {/* Class distribution */}
          <div className="lg:col-span-3 bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-5">
              Prediction Breakdown
            </h2>
            {stats.classCounts.length === 0 ? (
              <p className="text-sm text-slate-400">No data yet.</p>
            ) : (
              <ul className="space-y-3">
                {stats.classCounts.map(c => (
                  <li key={c.id}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium text-slate-700 truncate pr-3">{c.name}</span>
                      <span className="text-xs font-bold text-slate-500 tabular-nums shrink-0">{c.count}</span>
                    </div>
                    <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
                      <div
                        className={['h-full rounded-full transition-all duration-500', BAR_COLORS[c.id] ?? 'bg-teal-400'].join(' ')}
                        style={{ width: `${Math.round((c.count / maxCount) * 100)}%` }}
                      />
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Right column: last analysis + actions */}
          <div className="lg:col-span-2 flex flex-col gap-4">

            {/* Last analysis */}
            {stats.lastAnalysis && (
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
                <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-4">
                  Last Analysis
                </h2>
                <div className="flex items-start gap-4">
                  {/* Thumbnail */}
                  <div className="w-14 h-14 rounded-xl bg-slate-100 border border-slate-200 overflow-hidden shrink-0 flex items-center justify-center text-slate-300">
                    {stats.lastAnalysis.imageUrl ? (
                      <img
                        src={stats.lastAnalysis.imageUrl}
                        alt="Last analysis"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    )}
                  </div>

                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-slate-800 truncate leading-none mb-1">
                      {stats.lastAnalysis.className}
                    </p>
                    <div className="flex items-center gap-2 mb-2">
                      <span className={[
                        'text-[10px] font-bold px-1.5 py-0.5 rounded border',
                        RISK_LABEL[stats.lastAnalysis.classId]?.cls ?? 'text-slate-500 bg-slate-50 border-slate-200',
                      ].join(' ')}>
                        {RISK_LABEL[stats.lastAnalysis.classId]?.label ?? 'Unknown'}
                      </span>
                      <span className="text-xs text-slate-400 font-medium tabular-nums">
                        {stats.lastAnalysis.confidence.toFixed(1)}% conf.
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400">{stats.lastAnalysis.date}</p>
                  </div>
                </div>

                <button
                  onClick={onNavigateToHistory}
                  className="mt-4 w-full text-xs font-semibold text-teal-600 hover:text-teal-700 text-right flex items-center justify-end gap-1 transition-colors"
                >
                  View all records
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </div>
            )}

            {/* Actions */}
            <div className="flex flex-col gap-3">
              <Button onClick={onNavigateToUpload}>
                <div className="flex items-center justify-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                  New Analysis
                </div>
              </Button>
              <Button variant="secondary" onClick={onNavigateToHistory}>
                <div className="flex items-center justify-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 002 2h2a2 2 0 002-2" />
                  </svg>
                  View History
                </div>
              </Button>
            </div>

          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-6 text-center bg-slate-50">
        <p className="text-[11px] font-medium text-slate-400 uppercase tracking-widest px-4">
          Designed to assist medical professionals. Not a replacement for clinical diagnosis.
        </p>
      </footer>
    </div>
  );
};

export default DashboardScreen;
