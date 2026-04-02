import React from 'react';

export interface BarData {
  id: string;
  label: string;
  value: number;
  color?: string;
}

interface VerticalBarChartProps {
  data: BarData[];
  title: string;
  subtitle?: string;
  showValues?: boolean;
  maxBars?: number;
  height?: number;
}

// Gentle, eye-friendly color palette - consistent across the app
export const GENTLE_COLORS = {
  // Condition-specific colors (softer versions)
  mel: '#f87171',    // Soft rose (critical)
  bcc: '#fb923c',    // Soft orange (high)
  akiec: '#fbbf24',  // Soft amber (moderate)
  bkl: '#34d399',    // Soft emerald (low)
  df: '#22d3ee',     // Soft cyan (low)
  nv: '#a78bfa',     // Soft violet (low)
  vasc: '#f472b6',   // Soft pink (low)

  // Generic palette for charts
  palette: [
    '#6ee7b7', // Soft mint
    '#7dd3fc', // Soft sky
    '#c4b5fd', // Soft lavender
    '#fda4af', // Soft rose
    '#fcd34d', // Soft gold
    '#a5f3fc', // Soft aqua
    '#d9f99d', // Soft lime
  ],
};

const VerticalBarChart: React.FC<VerticalBarChartProps> = ({
  data,
  title,
  subtitle,
  showValues = true,
  maxBars = 7,
  height = 200,
}) => {
  if (data.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-slate-300 p-6">
        <h3 className="text-sm font-bold text-slate-700 mb-1">{title}</h3>
        {subtitle && <p className="text-xs text-slate-400 mb-4">{subtitle}</p>}
        <div className="flex items-center justify-center text-slate-400 text-sm" style={{ height }}>
          No data available
        </div>
      </div>
    );
  }

  const displayData = data.slice(0, maxBars);
  const maxValue = Math.max(...displayData.map((d) => d.value), 1);

  return (
    <div className="bg-white rounded-xl border border-slate-300 p-6">
      <h3 className="text-sm font-bold text-slate-700 mb-1">{title}</h3>
      {subtitle && <p className="text-xs text-slate-400 mb-4">{subtitle}</p>}

      <div className="flex items-end justify-between gap-2" style={{ height }}>
        {displayData.map((item, index) => {
          const barHeight = maxValue > 0 ? (item.value / maxValue) * 100 : 0;
          const color = item.color || GENTLE_COLORS.palette[index % GENTLE_COLORS.palette.length];

          return (
            <div
              key={item.id}
              className="flex-1 flex flex-col items-center justify-end h-full min-w-0"
            >
              {/* Value label */}
              {showValues && item.value > 0 && (
                <span className="text-xs font-semibold text-slate-600 mb-1 tabular-nums">
                  {item.value}
                </span>
              )}

              {/* Bar */}
              <div
                className="w-full max-w-[40px] rounded-t-lg transition-all duration-500 hover:opacity-80"
                style={{
                  height: `${Math.max(barHeight, 4)}%`,
                  backgroundColor: color,
                  minHeight: item.value > 0 ? '8px' : '2px',
                }}
              />

              {/* Label */}
              <div className="mt-2 w-full text-center">
                <span className="text-[10px] font-medium text-slate-500 leading-tight block truncate px-0.5">
                  {item.label}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default VerticalBarChart;
