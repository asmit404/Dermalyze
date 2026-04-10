import type { AnalysisHistoryItem } from './types';
import {
  CLASS_DEFINITIONS,
  CLASSES_BY_RISK,
  RISK_COLORS,
  HIGH_RISK_CLASS_IDS,
  isValidClassId,
  type RiskLevel,
} from './classDefinitions';

export interface TimeSeriesData {
  date: string;
  count: number;
  avgConfidence: number;
}

export interface RiskDistribution {
  level: string;
  count: number;
  color: string;
}

export interface HeatmapData {
  day: number; // 0-6 (Sun-Sat)
  hour: number; // 0-23
  count: number;
}

export interface DiagnosisData {
  id: string;
  name: string;
  count: number;
  percentage: number;
  color: string;
  riskLevel: RiskLevel;
}

/**
 * Aggregate analyses by time period (day, week, or month)
 */
export function aggregateByTimePeriod(
  items: AnalysisHistoryItem[],
  period: 'day' | 'week' | 'month',
  months: number
): TimeSeriesData[] {
  const now = new Date();
  const startDate = new Date(now.getTime() - months * 30 * 24 * 60 * 60 * 1000);

  // Create map to aggregate data
  const dataMap = new Map<string, { count: number; totalConf: number }>();

  items.forEach((item) => {
    const date = new Date(item.createdAt); // Use raw timestamp
    if (date < startDate) return;

    let key: string;
    if (period === 'day') {
      key = date.toISOString().split('T')[0];
    } else if (period === 'week') {
      const weekNum = Math.floor(date.getTime() / (7 * 24 * 60 * 60 * 1000));
      key = `week-${weekNum}`;
    } else {
      key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
    }

    const existing = dataMap.get(key) || { count: 0, totalConf: 0 };
    dataMap.set(key, {
      count: existing.count + 1,
      totalConf: existing.totalConf + item.confidence,
    });
  });

  // Convert map to sorted array
  return Array.from(dataMap.entries())
    .map(([date, data]) => ({
      date,
      count: data.count,
      avgConfidence: data.count > 0 ? data.totalConf / data.count : 0,
    }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

/**
 * Calculate risk level distribution
 */
export function getRiskDistribution(items: AnalysisHistoryItem[]): RiskDistribution[] {
  const riskConfig: Record<RiskLevel, { label: string }> = {
    critical: { label: 'Critical' },
    high: { label: 'High' },
    moderate: { label: 'Moderate' },
    low: { label: 'Low' },
  };

  const distribution: RiskDistribution[] = [];

  (Object.keys(riskConfig) as RiskLevel[]).forEach((level) => {
    const classIds = CLASSES_BY_RISK[level];
    const count = items.filter(
      (item) => isValidClassId(item.classId) && classIds.includes(item.classId)
    ).length;
    if (count > 0) {
      distribution.push({
        level: riskConfig[level].label,
        count,
        color: RISK_COLORS[level],
      });
    }
  });

  return distribution;
}

/**
 * Generate heatmap data (day of week x hour of day)
 */
export function getHeatmapData(items: AnalysisHistoryItem[]): HeatmapData[] {
  const heatmap = new Map<string, number>();

  items.forEach((item) => {
    const date = new Date(item.createdAt); // Use raw timestamp
    const day = date.getDay(); // 0-6
    const hour = date.getHours(); // 0-23
    const key = `${day}-${hour}`;
    heatmap.set(key, (heatmap.get(key) || 0) + 1);
  });

  const data: HeatmapData[] = [];
  for (let day = 0; day < 7; day++) {
    for (let hour = 0; hour < 24; hour++) {
      const key = `${day}-${hour}`;
      data.push({
        day,
        hour,
        count: heatmap.get(key) || 0,
      });
    }
  }

  return data;
}

/**
 * Get summary statistics
 */
export function getSummaryStats(items: AnalysisHistoryItem[]) {
  if (items.length === 0) {
    return {
      total: 0,
      avgConfidence: 0,
      highRiskCount: 0,
      lastAnalysisDate: null,
    };
  }

  const avgConfidence = items.reduce((sum, item) => sum + item.confidence, 0) / items.length;

  const highRiskCount = items.filter(
    (item) => isValidClassId(item.classId) && HIGH_RISK_CLASS_IDS.includes(item.classId)
  ).length;

  // Find most recent analysis using raw timestamps
  const dates = items.map((item) => new Date(item.createdAt));
  const lastAnalysisDate = new Date(Math.max(...dates.map((d) => d.getTime())));

  return {
    total: items.length,
    avgConfidence: Math.round(avgConfidence * 10) / 10,
    highRiskCount,
    lastAnalysisDate: lastAnalysisDate.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    }),
  };
}

// Re-export CLASS_COLORS for backward compatibility
export { CLASS_COLORS as GENTLE_CONDITION_COLORS } from './classDefinitions';

export interface PredictionBreakdownData {
  id: string;
  label: string;
  value: number;
  color: string;
}

/**
 * Get prediction breakdown aggregated by week or month
 */
export function getPredictionBreakdown(
  items: AnalysisHistoryItem[],
  period: 'weekly' | 'monthly'
): PredictionBreakdownData[] {
  if (items.length === 0) return [];

  const now = new Date();
  const periodCount = period === 'weekly' ? 8 : 6; // Last 8 weeks or 6 months

  // Create period labels and counts
  const periods: { start: Date; end: Date; label: string }[] = [];

  if (period === 'weekly') {
    // Get last 8 weeks
    for (let i = periodCount - 1; i >= 0; i--) {
      const end = new Date(now);
      end.setDate(end.getDate() - i * 7);
      end.setHours(23, 59, 59, 999);

      const start = new Date(end);
      start.setDate(start.getDate() - 6);
      start.setHours(0, 0, 0, 0);

      const weekLabel = start.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      periods.push({ start, end, label: weekLabel });
    }
  } else {
    // Get last 6 months
    for (let i = periodCount - 1; i >= 0; i--) {
      const monthDate = new Date(now.getFullYear(), now.getMonth() - i, 1);
      const start = new Date(monthDate.getFullYear(), monthDate.getMonth(), 1);
      const end = new Date(monthDate.getFullYear(), monthDate.getMonth() + 1, 0, 23, 59, 59, 999);

      const monthLabel = start.toLocaleDateString('en-US', { month: 'short' });
      periods.push({ start, end, label: monthLabel });
    }
  }

  // Count items in each period
  const palette = [
    '#6ee7b7',
    '#7dd3fc',
    '#c4b5fd',
    '#fda4af',
    '#fcd34d',
    '#a5f3fc',
    '#d9f99d',
    '#fed7aa',
  ];

  return periods.map((p, index) => {
    const count = items.filter((item) => {
      const date = new Date(item.createdAt);
      return date >= p.start && date <= p.end;
    }).length;

    return {
      id: `period-${index}`,
      label: p.label,
      value: count,
      color: palette[index % palette.length],
    };
  });
}

/**
 * Get diagnosis breakdown by condition type
 * Returns counts and percentages for each lesion type, sorted by risk level
 */
export function getDiagnosisBreakdown(items: AnalysisHistoryItem[]): DiagnosisData[] {
  if (items.length === 0) return [];

  // Count occurrences of each condition
  const countMap = new Map<string, number>();
  items.forEach((item) => {
    const current = countMap.get(item.classId) || 0;
    countMap.set(item.classId, current + 1);
  });

  // Build diagnosis data with clinical info from single source of truth
  const diagnoses: DiagnosisData[] = [];
  countMap.forEach((count, id) => {
    if (!isValidClassId(id)) return;

    const classDef = CLASS_DEFINITIONS[id];
    diagnoses.push({
      id,
      name: classDef.name,
      count,
      percentage: Math.round((count / items.length) * 1000) / 10,
      color: classDef.color,
      riskLevel: classDef.riskLevel,
    });
  });

  // Sort by risk level (critical first), then by count
  const riskOrder: Record<RiskLevel, number> = { critical: 0, high: 1, moderate: 2, low: 3 };
  return diagnoses.sort((a, b) => {
    const riskDiff = riskOrder[a.riskLevel] - riskOrder[b.riskLevel];
    if (riskDiff !== 0) return riskDiff;
    return b.count - a.count;
  });
}
