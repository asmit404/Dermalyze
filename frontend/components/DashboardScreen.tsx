import React, { useEffect, useState } from 'react';
import Button from './ui/Button';
import { useDataCache } from '../lib/dataCache';
import { getPredictionBreakdown } from '../lib/analyticsUtils';
import { supabase } from '../lib/supabase';
import type { AnalysisHistoryItem } from '../lib/types';

interface DashboardScreenProps {
  onNavigateToUpload: () => void;
  onNavigateToHistory: () => void;
  onNavigateToTrends: () => void;
}

// Classes that warrant clinical follow-up — kept for reference, filtering done server-side via RPC
const RISK_LABEL: Record<string, { label: string; cls: string }> = {
  mel: { label: 'Critical', cls: 'text-red-600 bg-red-50 border-red-200' },
  bcc: { label: 'High', cls: 'text-orange-600 bg-orange-50 border-orange-200' },
  akiec: { label: 'Moderate', cls: 'text-amber-600 bg-amber-50 border-amber-200' },
  bkl: { label: 'Low', cls: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
  df: { label: 'Low', cls: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
  nv: { label: 'Low', cls: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
  vasc: { label: 'Low', cls: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
};

const DashboardScreen: React.FC<DashboardScreenProps> = ({
  onNavigateToUpload,
  onNavigateToHistory,
  onNavigateToTrends,
}) => {
  const { dashboardStats, fetchDashboardStats, userName } = useDataCache();
  const { data: stats, loading, error: fetchErr } = dashboardStats;
  const [timePeriod, setTimePeriod] = useState<'weekly' | 'monthly'>('weekly');

  // Analytics data - fetch ALL records (not just first page)
  const [analyticsData, setAnalyticsData] = useState<AnalysisHistoryItem[]>([]);
  const [analyticsLoading, setAnalyticsLoading] = useState(true);

  useEffect(() => {
    fetchDashboardStats();
  }, [fetchDashboardStats]);

  // Fetch all analytics data for charts
  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        setAnalyticsLoading(true);

        // Get user directly from auth (same pattern as dashboard stats)
        const {
          data: { user },
        } = await supabase.auth.getUser();
        if (!user) {
          setAnalyticsData([]);
          setAnalyticsLoading(false);
          return;
        }

        const { data, error } = await supabase
          .from('analyses')
          .select('id, created_at, predicted_class_id, predicted_class_name, confidence')
          .eq('user_id', user.id)
          .order('created_at', { ascending: false });

        if (error) throw error;

        const mappedData = (data ?? []).map((row) => ({
          id: row.id,
          createdAt: row.created_at,
          date: '',
          time: '',
          classId: row.predicted_class_id,
          className: row.predicted_class_name,
          confidence: row.confidence,
        }));

        setAnalyticsData(mappedData);
      } catch (err) {
        console.error('Failed to fetch analytics data:', err);
        setAnalyticsData([]);
      } finally {
        setAnalyticsLoading(false);
      }
    };

    fetchAnalytics();
  }, []); // Run once on mount, user is fetched inside

  // Get prediction breakdown data for the selected time period
  const predictionBreakdownData = getPredictionBreakdown(analyticsData, timePeriod);

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
              <div
                key={i}
                className="h-24 bg-white border border-slate-200 rounded-2xl animate-pulse"
              />
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
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
              </div>
              <h2 className="text-2xl font-semibold text-slate-900 mb-3 tracking-wide">
                {greeting()}
                {userName ? `, ${userName}.` : ''}
              </h2>
              <p className="text-slate-500 text-base mb-8 max-w-sm mx-auto leading-relaxed">
                No analyses yet. Upload your first dermatoscopic image to begin.
              </p>
              <Button onClick={onNavigateToUpload}>
                <div className="flex items-center justify-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                    />
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
            {greeting()}
            {userName ? `, ${userName}` : ''}
          </h1>
          <p className="text-sm text-slate-400 mt-0.5">Clinical analysis summary</p>
        </div>

        {/* ── Stat strip ── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Total */}
          <div className="bg-white rounded-2xl border border-slate-300 shadow-sm p-5">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">
              Total Analyses
            </p>
            <p className="text-3xl font-bold text-slate-900 tracking-tight">{stats.total}</p>
          </div>

          {/* This month */}
          <div className="bg-white rounded-2xl border border-slate-300 shadow-sm p-5">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">
              This Month
            </p>
            <p className="text-3xl font-bold text-teal-600 tracking-tight">{stats.thisMonth}</p>
          </div>

          {/* Avg confidence */}
          <div className="bg-white rounded-2xl border border-slate-300 shadow-sm p-5">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">
              Avg Confidence
            </p>
            <p
              className={[
                'text-3xl font-bold tracking-tight',
                stats.avgConfidence !== null && stats.avgConfidence >= 80
                  ? 'text-emerald-600'
                  : stats.avgConfidence !== null && stats.avgConfidence >= 50
                    ? 'text-amber-600'
                    : 'text-red-600',
              ].join(' ')}
            >
              {stats.avgConfidence !== null ? `${stats.avgConfidence}%` : '—'}
            </p>
          </div>

          {/* Needs review */}
          <div
            className={[
              'rounded-2xl border shadow-sm p-5',
              stats.needsReview > 0 ? 'bg-amber-50 border-amber-300' : 'bg-white border-slate-200',
            ].join(' ')}
          >
            <p
              className={[
                'text-[10px] font-bold uppercase tracking-widest mb-1',
                stats.needsReview > 0 ? 'text-amber-600' : 'text-slate-400',
              ].join(' ')}
            >
              Needs Review
            </p>
            <p
              className={[
                'text-3xl font-bold tracking-tight',
                stats.needsReview > 0 ? 'text-amber-700' : 'text-slate-900',
              ].join(' ')}
            >
              {stats.needsReview}
            </p>
            <p className="text-[10px] text-amber-500 mt-0.5 font-medium">
              {stats.needsReview > 0 ? 'MEL · BCC · AK' : 'No flags'}
            </p>
          </div>
        </div>

        {/* ── Main grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Prediction breakdown - Vertical Bar Chart */}
          <div className="lg:col-span-3 bg-white rounded-2xl border border-slate-300 shadow-sm p-6">
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest">
                Prediction Breakdown
              </h2>
              {/* Weekly/Monthly Toggle */}
              <div className="flex items-center gap-1 bg-slate-300 rounded-lg p-0.5">
                <button
                  onClick={() => setTimePeriod('weekly')}
                  className={[
                    'px-3 py-1 text-xs font-semibold rounded-md transition-colors',
                    timePeriod === 'weekly'
                      ? 'bg-white text-slate-700 shadow-sm'
                      : 'text-slate-500 hover:text-slate-700',
                  ].join(' ')}
                >
                  Weekly
                </button>
                <button
                  onClick={() => setTimePeriod('monthly')}
                  className={[
                    'px-3 py-1 text-xs font-semibold rounded-md transition-colors',
                    timePeriod === 'monthly'
                      ? 'bg-white text-slate-700 shadow-sm'
                      : 'text-slate-500 hover:text-slate-700',
                  ].join(' ')}
                >
                  Monthly
                </button>
              </div>
            </div>

            {analyticsLoading ? (
              <div className="flex items-center justify-center h-48">
                <div className="flex flex-col items-center gap-2">
                  <div className="w-8 h-8 border-3 border-slate-200 border-t-teal-500 rounded-full animate-spin" />
                  <span className="text-xs text-slate-400">Loading chart...</span>
                </div>
              </div>
            ) : predictionBreakdownData.length === 0 ? (
              <div className="flex items-center justify-center h-48 text-slate-400 text-sm">
                No data yet.
              </div>
            ) : (
              <div className="flex items-end justify-between gap-3" style={{ height: 180 }}>
                {predictionBreakdownData.map((item) => {
                  const maxValue = Math.max(...predictionBreakdownData.map((d) => d.value), 1);
                  const barHeight = maxValue > 0 ? (item.value / maxValue) * 100 : 0;

                  return (
                    <div
                      key={item.id}
                      className="flex-1 flex flex-col items-center justify-end h-full min-w-0"
                    >
                      {/* Value label */}
                      {item.value > 0 && (
                        <span className="text-xs font-semibold text-slate-600 mb-1 tabular-nums">
                          {item.value}
                        </span>
                      )}

                      {/* Bar */}
                      <div
                        className="w-full max-w-[36px] rounded-t-lg transition-all duration-500 hover:opacity-80"
                        style={{
                          height: `${Math.max(barHeight, 4)}%`,
                          backgroundColor: item.color,
                          minHeight: item.value > 0 ? '8px' : '2px',
                        }}
                      />

                      {/* Label */}
                      <div className="mt-2 w-full text-center">
                        <span className="text-[10px] font-medium text-slate-500 leading-tight block truncate">
                          {item.label}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Right column: last analysis + actions */}
          <div className="lg:col-span-2 flex flex-col gap-4">
            {/* Last analysis */}
            {stats.lastAnalysis && (
              <div className="bg-white rounded-2xl border border-slate-300 shadow-sm p-5">
                <h2 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-4">
                  Last Analysis
                </h2>
                <div className="flex items-start gap-4">
                  {/* Thumbnail */}
                  <div className="w-14 h-14 rounded-xl bg-slate-200 border border-slate-200 overflow-hidden shrink-0 flex items-center justify-center text-slate-300">
                    {stats.lastAnalysis.imageUrl ? (
                      <img
                        src={stats.lastAnalysis.imageUrl}
                        alt="Last analysis"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <svg
                        className="w-6 h-6"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={1.5}
                          d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                        />
                      </svg>
                    )}
                  </div>

                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-slate-800 truncate leading-none mb-1">
                      {stats.lastAnalysis.className}
                    </p>
                    <div className="flex items-center gap-2 mb-2">
                      <span
                        className={[
                          'text-[10px] font-bold px-1.5 py-0.5 rounded border',
                          RISK_LABEL[stats.lastAnalysis.classId]?.cls ??
                            'text-slate-500 bg-slate-50 border-slate-200',
                        ].join(' ')}
                      >
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
                  <svg
                    className="w-3.5 h-3.5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                </button>
              </div>
            )}

            {/* Actions */}
            <div className="flex flex-col gap-3">
              <Button onClick={onNavigateToUpload}>
                <div className="flex items-center justify-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                    />
                  </svg>
                  New Analysis
                </div>
              </Button>
              <Button variant="secondary" onClick={onNavigateToHistory}>
                <div className="flex items-center justify-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 002 2h2a2 2 0 002-2"
                    />
                  </svg>
                  View History
                </div>
              </Button>
              <Button variant="secondary" onClick={onNavigateToTrends}>
                <div className="flex items-center justify-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                    />
                  </svg>
                  View Trends
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
