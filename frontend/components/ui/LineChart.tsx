import React, { useState } from 'react';

interface DataPoint {
  date: string;
  count: number;
  avgConfidence?: number;
}

interface LineChartProps {
  data: DataPoint[];
  title: string;
  color?: string;
  height?: number;
  showConfidence?: boolean;
}

// Helper function to format date labels based on format
const formatDateLabel = (dateStr: string): string => {
  // Week format: "week-12345"
  if (dateStr.startsWith('week-')) {
    const weekNum = parseInt(dateStr.split('-')[1], 10);
    // Convert week number to approximate date
    const date = new Date(weekNum * 7 * 24 * 60 * 60 * 1000);
    return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  }

  // Month format: "2026-01"
  if (dateStr.match(/^\d{4}-\d{2}$/)) {
    const date = new Date(`${dateStr}-01`);
    return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  }

  // Day format: "2026-01-02" (ISO date)
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
};

const LineChart: React.FC<LineChartProps> = ({
  data,
  title,
  color = '#0d9488',
  height = 200,
  showConfidence = false,
}) => {
  const [hoveredPoint, setHoveredPoint] = useState<number | null>(null);

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-slate-400 text-sm">
        No data available
      </div>
    );
  }

  const maxCount = Math.max(...data.map((d) => d.count));
  const padding = 40;
  const width = 600;
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;

  // Generate path for line
  const points = data.map((d, i) => {
    const x = padding + (i / Math.max(data.length - 1, 1)) * chartWidth;
    const y = padding + chartHeight - (d.count / Math.max(maxCount, 1)) * chartHeight;
    return { x, y, data: d };
  });

  const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

  // Area fill path
  const areaPath =
    linePath +
    ` L ${points[points.length - 1].x} ${padding + chartHeight} L ${points[0].x} ${
      padding + chartHeight
    } Z`;

  // Y-axis labels
  const yLabels = [0, Math.floor(maxCount / 2), maxCount];

  return (
    <div className="bg-white rounded-xl border border-slate-300 p-6">
      <h3 className="text-sm font-bold text-slate-700 mb-4">{title}</h3>
      <div className="relative">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full"
          style={{ maxWidth: '100%', height: 'auto' }}
        >
          {/* Grid lines */}
          {yLabels.map((label) => {
            const y = padding + chartHeight - (label / Math.max(maxCount, 1)) * chartHeight;
            return (
              <g key={label}>
                <line
                  x1={padding}
                  y1={y}
                  x2={width - padding}
                  y2={y}
                  stroke="#e2e8f0"
                  strokeWidth="1"
                />
                <text x={padding - 10} y={y + 4} fontSize="10" fill="#94a3b8" textAnchor="end">
                  {label}
                </text>
              </g>
            );
          })}

          {/* Area fill */}
          <path d={areaPath} fill={color} fillOpacity="0.1" />

          {/* Line */}
          <path d={linePath} stroke={color} strokeWidth="2" fill="none" />

          {/* Points */}
          {points.map((p, i) => (
            <g key={i}>
              <circle
                cx={p.x}
                cy={p.y}
                r={hoveredPoint === i ? 6 : 4}
                fill={color}
                className="cursor-pointer transition-all"
                onMouseEnter={() => setHoveredPoint(i)}
                onMouseLeave={() => setHoveredPoint(null)}
              />
              {/* Larger invisible hit area for easier hovering */}
              <circle
                cx={p.x}
                cy={p.y}
                r="12"
                fill="transparent"
                className="cursor-pointer"
                onMouseEnter={() => setHoveredPoint(i)}
                onMouseLeave={() => setHoveredPoint(null)}
              />
            </g>
          ))}

          {/* X-axis labels (show first, middle, last) */}
          {[0, Math.floor(data.length / 2), data.length - 1].map((i) => {
            if (i >= data.length) return null;
            const p = points[i];
            return (
              <text
                key={i}
                x={p.x}
                y={padding + chartHeight + 20}
                fontSize="10"
                fill="#94a3b8"
                textAnchor="middle"
              >
                {formatDateLabel(data[i].date)}
              </text>
            );
          })}
        </svg>

        {/* Custom tooltip */}
        {hoveredPoint !== null && (
          <div
            className="absolute bg-slate-800 text-white text-xs rounded-lg px-3 py-2 pointer-events-none shadow-lg z-10"
            style={{
              left: `${(points[hoveredPoint].x / width) * 100}%`,
              top: `${(points[hoveredPoint].y / height) * 100 - 15}%`,
              transform: 'translate(-50%, -100%)',
            }}
          >
            <div className="font-medium">{formatDateLabel(data[hoveredPoint].date)}</div>
            <div className="text-slate-300">
              {data[hoveredPoint].count} {data[hoveredPoint].count === 1 ? 'analysis' : 'analyses'}
            </div>
            {showConfidence && data[hoveredPoint].avgConfidence !== undefined && (
              <div className="text-teal-300 font-medium">
                {data[hoveredPoint].avgConfidence.toFixed(1)}% avg confidence
              </div>
            )}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-4 mt-4 text-xs text-slate-600">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
          <span>Analyses</span>
        </div>
        {showConfidence && (
          <span className="text-slate-400">• Hover points for details</span>
        )}
      </div>
    </div>
  );
};

export default LineChart;
