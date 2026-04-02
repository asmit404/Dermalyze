import React, { useState } from 'react';

interface DiagnosisData {
  id: string;
  name: string;
  count: number;
  percentage: number;
  color: string;
  riskLevel: 'critical' | 'high' | 'moderate' | 'low';
}

interface DiagnosisChartProps {
  data: DiagnosisData[];
  title: string;
}

const RISK_BADGES: Record<string, { label: string; bg: string; text: string }> = {
  critical: { label: 'Critical', bg: 'bg-red-50', text: 'text-rose-600' },
  high: { label: 'High', bg: 'bg-orange-50', text: 'text-orange-600' },
  moderate: { label: 'Moderate', bg: 'bg-amber-50', text: 'text-amber-600' },
  low: { label: 'Low', bg: 'bg-emerald-50', text: 'text-emerald-600' },
};

const DiagnosisChart: React.FC<DiagnosisChartProps> = ({ data, title }) => {
  const [viewMode, setViewMode] = useState<'bar' | 'pie'>('bar');

  if (data.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-slate-300 p-6">
        <h3 className="text-sm font-bold text-slate-700 mb-4">{title}</h3>
        <div className="flex items-center justify-center h-48 text-slate-400 text-sm">
          No diagnosis data available
        </div>
      </div>
    );
  }

  const maxCount = Math.max(...data.map((d) => d.count));
  const total = data.reduce((sum, d) => sum + d.count, 0);

  // Calculate pie chart segments
  const renderPieChart = () => {
    let cumulativePercentage = 0;
    const size = 160;
    const radius = 70;
    const center = size / 2;

    return (
      <div className="flex items-center justify-center gap-8">
        {/* Pie */}
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          {data.map((item, index) => {
            const percentage = total > 0 ? (item.count / total) * 100 : 0;
            const startAngle = (cumulativePercentage / 100) * 2 * Math.PI - Math.PI / 2;
            cumulativePercentage += percentage;
            const endAngle = (cumulativePercentage / 100) * 2 * Math.PI - Math.PI / 2;

            // Calculate path
            const x1 = center + radius * Math.cos(startAngle);
            const y1 = center + radius * Math.sin(startAngle);
            const x2 = center + radius * Math.cos(endAngle);
            const y2 = center + radius * Math.sin(endAngle);
            const largeArcFlag = percentage > 50 ? 1 : 0;

            // Handle full circle case
            if (percentage >= 99.9) {
              return (
                <circle
                  key={item.id}
                  cx={center}
                  cy={center}
                  r={radius}
                  fill={item.color}
                  className="transition-opacity duration-300 hover:opacity-80"
                />
              );
            }

            const pathData = `
              M ${center} ${center}
              L ${x1} ${y1}
              A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2}
              Z
            `;

            return (
              <path
                key={item.id}
                d={pathData}
                fill={item.color}
                className="transition-opacity duration-300 hover:opacity-80"
              />
            );
          })}
          {/* Center hole for donut effect */}
          <circle cx={center} cy={center} r={40} fill="white" />
          <text
            x={center}
            y={center}
            textAnchor="middle"
            dy="0.3em"
            className="text-lg font-bold fill-slate-700"
          >
            {total}
          </text>
          <text
            x={center}
            y={center + 16}
            textAnchor="middle"
            className="text-[10px] fill-slate-400"
          >
            total
          </text>
        </svg>

        {/* Legend */}
        <div className="space-y-2">
          {data.map((item) => {
            const badge = RISK_BADGES[item.riskLevel];
            return (
              <div key={item.id} className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full shrink-0"
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-xs font-medium text-slate-600 truncate max-w-[100px]">
                  {item.name.split(' ')[0]}
                </span>
                <span className="text-xs font-bold text-slate-800 tabular-nums">
                  {item.count}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  // Render vertical bar chart
  const renderBarChart = () => (
    <div className="space-y-4">
      <div className="flex items-end justify-between gap-2" style={{ height: 160 }}>
        {data.map((item) => {
          const percentage = maxCount > 0 ? (item.count / maxCount) * 100 : 0;

          return (
            <div
              key={item.id}
              className="flex-1 flex flex-col items-center justify-end h-full min-w-0"
            >
              {/* Value label */}
              {item.count > 0 && (
                <span className="text-xs font-semibold text-slate-600 mb-1 tabular-nums">
                  {item.count}
                </span>
              )}

              {/* Bar */}
              <div
                className="w-full max-w-[32px] rounded-t-lg transition-all duration-500 hover:opacity-80"
                style={{
                  height: `${Math.max(percentage, 6)}%`,
                  backgroundColor: item.color,
                  minHeight: item.count > 0 ? '8px' : '2px',
                }}
              />

              {/* Label - shortened */}
              <div className="mt-2 w-full text-center">
                <span className="text-[9px] font-medium text-slate-500 leading-tight block truncate px-0.5">
                  {item.name === 'Melanocytic Nevi' ? 'Nevi' :
                   item.name === 'Basal Cell Carcinoma' ? 'BCC' :
                   item.name === 'Actinic Keratoses' ? 'AK' :
                   item.name === 'Benign Keratosis' ? 'BK' :
                   item.name === 'Dermatofibroma' ? 'DF' :
                   item.name === 'Vascular Lesions' ? 'Vasc' :
                   item.name === 'Melanoma' ? 'Mel' :
                   item.name.slice(0, 4)}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Risk level legend */}
      <div className="pt-3 border-t border-slate-100">
        <div className="flex flex-wrap items-center justify-center gap-2 text-[10px]">
          {Object.entries(RISK_BADGES).map(([key, badge]) => (
            <div key={key} className="flex items-center gap-1">
              <span className={`px-1.5 py-0.5 rounded ${badge.bg} ${badge.text} font-semibold`}>
                {badge.label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  return (
    <div className="bg-white rounded-xl border border-slate-300 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-bold text-slate-700">{title}</h3>

        {/* Bar/Pie Toggle */}
        <div className="flex items-center gap-1 bg-slate-100 rounded-lg p-0.5">
          <button
            onClick={() => setViewMode('bar')}
            className={[
              'p-1.5 rounded-md transition-colors',
              viewMode === 'bar'
                ? 'bg-white text-slate-700 shadow-sm'
                : 'text-slate-500 hover:text-slate-700',
            ].join(' ')}
            title="Bar chart"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </button>
          <button
            onClick={() => setViewMode('pie')}
            className={[
              'p-1.5 rounded-md transition-colors',
              viewMode === 'pie'
                ? 'bg-white text-slate-700 shadow-sm'
                : 'text-slate-500 hover:text-slate-700',
            ].join(' ')}
            title="Pie chart"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
            </svg>
          </button>
        </div>
      </div>

      {viewMode === 'bar' ? renderBarChart() : renderPieChart()}
    </div>
  );
};

export default DiagnosisChart;
