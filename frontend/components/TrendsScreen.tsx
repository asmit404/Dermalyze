import React, { useState, useEffect } from 'react';
import Button from './ui/Button';
import LineChart from './ui/LineChart';
import PieChart from './ui/PieChart';
import DiagnosisChart from './ui/DiagnosisChart';
import Heatmap from './ui/Heatmap';
import { supabase } from '../lib/supabase';
import {
  aggregateByTimePeriod,
  getRiskDistribution,
  getDiagnosisBreakdown,
  getHeatmapData,
  getSummaryStats,
} from '../lib/analyticsUtils';
import type { AnalysisHistoryItem } from '../lib/types';

interface TrendsScreenProps {
  onBack: () => void;
}

const TrendsScreen: React.FC<TrendsScreenProps> = ({ onBack }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [historyItems, setHistoryItems] = useState<AnalysisHistoryItem[]>([]);
  const [timePeriod, setTimePeriod] = useState<3 | 6 | 12>(3);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        const {
          data: { user },
        } = await supabase.auth.getUser();
        if (!user) throw new Error('Not authenticated');

        // Fetch all analyses for the user
        const { data, error: fetchError } = await supabase
          .from('analyses')
          .select(
            'id, created_at, predicted_class_id, predicted_class_name, confidence, image_url, all_scores, notes, trust_recommendation'
          )
          .eq('user_id', user.id)
          .order('created_at', { ascending: false });

        if (fetchError) throw fetchError;

        // Map to AnalysisHistoryItem format
        const items: AnalysisHistoryItem[] = (data || [])
          .filter((row) => row.trust_recommendation !== 'reject')
          .map((row) => ({
          id: row.id,
          createdAt: row.created_at, // Store raw ISO timestamp
          date: new Date(row.created_at).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
          }),
          time: new Date(row.created_at).toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
          }),
          classId: row.predicted_class_id,
          className: row.predicted_class_name,
          confidence: row.confidence,
          imageUrl: undefined,
          imagePath: undefined,
          allScores: row.all_scores as AnalysisHistoryItem['allScores'],
          notes: row.notes || undefined,
        }));

        setHistoryItems(items);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load analytics data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Filter items by time period
  const filteredItems = React.useMemo(() => {
    const now = new Date();
    const startDate = new Date(now.getTime() - timePeriod * 30 * 24 * 60 * 60 * 1000);
    return historyItems.filter((item) => new Date(item.createdAt) >= startDate);
  }, [historyItems, timePeriod]);

  // Calculate analytics using filtered data
  const timeSeriesData = aggregateByTimePeriod(filteredItems, 'day', timePeriod);
  const riskDistribution = getRiskDistribution(filteredItems);
  const diagnosisBreakdown = getDiagnosisBreakdown(filteredItems);
  const heatmapData = getHeatmapData(filteredItems);
  const summaryStats = getSummaryStats(filteredItems);

  return (
    <div className="flex-1 flex flex-col bg-slate-50 text-slate-900 pb-12">
      <main className="max-w-7xl mx-auto w-full px-4 sm:px-6 py-8 flex flex-col gap-8">
        {/* Header */}
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <button
              onClick={onBack}
              className="p-2 hover:bg-slate-300 rounded-full transition-colors text-slate-400 hover:text-slate-600"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 19l-7-7 7-7"
                />
              </svg>
            </button>
            <div>
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
                Analytics & Trends
              </h1>
              <p className="text-sm text-slate-500 mt-0.5">Insights from your analysis history</p>
            </div>
          </div>

          {/* Time Period Selector */}
          <div className="flex items-center gap-2 bg-white rounded-lg border border-slate-300 p-1">
            {[3, 6, 12].map((months) => (
              <button
                key={months}
                onClick={() => setTimePeriod(months as 3 | 6 | 12)}
                className={[
                  'px-3 py-1.5 text-xs font-bold rounded-md transition-colors',
                  timePeriod === months
                    ? 'bg-teal-600 text-white'
                    : 'text-slate-600 hover:bg-slate-50',
                ].join(' ')}
              >
                {months} months
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-32">
            <svg className="animate-spin h-8 w-8 text-teal-600" viewBox="0 0 24 24" fill="none">
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
          </div>
        ) : error ? (
          <div className="bg-white rounded-2xl border border-red-200 p-8 text-center">
            <svg
              className="w-12 h-12 text-red-500 mx-auto mb-3"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <p className="text-sm font-medium text-red-600">{error}</p>
          </div>
        ) : filteredItems.length === 0 ? (
          <div className="bg-white rounded-2xl border border-slate-300 p-16 text-center">
            <svg
              className="w-16 h-16 text-slate-300 mx-auto mb-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
              />
            </svg>
            <p className="text-slate-600 font-medium mb-2">
              {historyItems.length === 0
                ? 'No analysis data yet'
                : `No analyses in the last ${timePeriod} months`}
            </p>
            <p className="text-sm text-slate-400">
              {historyItems.length === 0
                ? 'Run some classifications to see your trends here'
                : 'Try selecting a longer time period'}
            </p>
          </div>
        ) : (
          <>
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
              <div className="bg-white rounded-xl border border-slate-300 p-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 bg-teal-50 rounded-lg">
                    <svg
                      className="w-6 h-6 text-teal-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                      />
                    </svg>
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Total Analyses
                    </p>
                    <p className="text-2xl font-bold text-slate-900 mt-0.5">{summaryStats.total}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-xl border border-slate-300 p-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 bg-blue-50 rounded-lg">
                    <svg
                      className="w-6 h-6 text-blue-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
                      />
                    </svg>
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Avg Confidence
                    </p>
                    <p className="text-2xl font-bold text-slate-900 mt-0.5">
                      {summaryStats.avgConfidence}%
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-xl border border-slate-300 p-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 bg-amber-50 rounded-lg">
                    <svg
                      className="w-6 h-6 text-amber-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                      />
                    </svg>
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      High Risk
                    </p>
                    <p className="text-2xl font-bold text-slate-900 mt-0.5">
                      {summaryStats.highRiskCount}
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-xl border border-slate-300 p-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 bg-slate-50 rounded-lg">
                    <svg
                      className="w-6 h-6 text-slate-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                      />
                    </svg>
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Last Analysis
                    </p>
                    <p className="text-sm font-bold text-slate-900 mt-0.5">
                      {summaryStats.lastAnalysisDate}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              {/* Time Series */}
              <LineChart
                data={timeSeriesData}
                title={`Analyses Over Time (Last ${timePeriod} Months)`}
                color="#0d9488"
                showConfidence
              />

              {/* Risk Distribution */}
              <PieChart data={riskDistribution} title="Risk Level Distribution" />

              {/* Diagnosis Breakdown */}
              <DiagnosisChart data={diagnosisBreakdown} title="Diagnosis Breakdown by Condition" />

              {/* Activity Heatmap */}
              <Heatmap data={heatmapData} title="Analysis Activity Pattern" />
            </div>

            {/* Insights Panel */}
            <div className="bg-gradient-to-br from-teal-50 to-blue-50 rounded-2xl border border-teal-300 p-6">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-white rounded-lg shadow-sm">
                  <svg
                    className="w-6 h-6 text-teal-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                    />
                  </svg>
                </div>
                <div className="flex-1">
                  <h3 className="text-sm font-bold text-slate-900 mb-2">Key Insights</h3>
                  <ul className="space-y-2 text-sm text-slate-700">
                    {summaryStats.highRiskCount > 0 && (
                      <li className="flex items-start gap-2">
                        <span className="text-amber-600 mt-0.5">•</span>
                        <span>
                          <strong>{summaryStats.highRiskCount}</strong> high-risk cases detected -
                          ensure proper follow-up
                        </span>
                      </li>
                    )}
                    {summaryStats.avgConfidence < 70 && (
                      <li className="flex items-start gap-2">
                        <span className="text-blue-600 mt-0.5">•</span>
                        <span>
                          Average confidence is <strong>{summaryStats.avgConfidence}%</strong> -
                          consider additional expert review
                        </span>
                      </li>
                    )}
                    {summaryStats.avgConfidence >= 85 && (
                      <li className="flex items-start gap-2">
                        <span className="text-emerald-600 mt-0.5">•</span>
                        <span>
                          High average confidence of <strong>{summaryStats.avgConfidence}%</strong>{' '}
                          indicates strong model certainty
                        </span>
                      </li>
                    )}
                    <li className="flex items-start gap-2">
                      <span className="text-teal-600 mt-0.5">•</span>
                      <span>
                        Analysis trends help identify usage patterns and optimize clinical workflows
                      </span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Back Button */}
        <div className="flex justify-center mt-4">
          <div className="max-w-xs w-full">
            <Button variant="secondary" onClick={onBack}>
              Back to Dashboard
            </Button>
          </div>
        </div>
      </main>

      <footer className="mt-auto py-10 text-center border-t border-slate-200 bg-white">
        <p className="text-[11px] font-medium text-slate-500 uppercase tracking-widest leading-relaxed">
          Analytics Dashboard. Clinical Support Tool for Medical Professionals.
        </p>
      </footer>
    </div>
  );
};

export default TrendsScreen;
