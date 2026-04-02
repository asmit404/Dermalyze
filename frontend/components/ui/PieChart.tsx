import React from 'react';

interface PieSlice {
  level: string;
  count: number;
  color: string;
}

interface PieChartProps {
  data: PieSlice[];
  title: string;
}

const PieChart: React.FC<PieChartProps> = ({ data, title }) => {
  if (data.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-slate-300 p-6">
        <h3 className="text-sm font-bold text-slate-700 mb-4">{title}</h3>
        <div className="flex items-center justify-center h-48 text-slate-400 text-sm">
          No data available
        </div>
      </div>
    );
  }

  const total = data.reduce((sum, d) => sum + d.count, 0);
  const size = 200;
  const center = size / 2;
  const radius = size / 2 - 10;

  // Calculate angles and create paths
  let currentAngle = -90; // Start at top
  const slices = data.map((slice) => {
    const percentage = (slice.count / total) * 100;
    const angle = (slice.count / total) * 360;
    const startAngle = currentAngle;
    const endAngle = currentAngle + angle;

    // Convert angles to radians
    const startRad = (startAngle * Math.PI) / 180;
    const endRad = (endAngle * Math.PI) / 180;

    // Calculate arc coordinates
    const x1 = center + radius * Math.cos(startRad);
    const y1 = center + radius * Math.sin(startRad);
    const x2 = center + radius * Math.cos(endRad);
    const y2 = center + radius * Math.sin(endRad);

    const largeArcFlag = angle > 180 ? 1 : 0;

    const path = [
      `M ${center} ${center}`,
      `L ${x1} ${y1}`,
      `A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2}`,
      'Z',
    ].join(' ');

    currentAngle = endAngle;

    return {
      ...slice,
      path,
      percentage: percentage.toFixed(1),
      midAngle: (startAngle + endAngle) / 2,
    };
  });

  return (
    <div className="bg-white rounded-xl border border-slate-300 p-6">
      <h3 className="text-sm font-bold text-slate-700 mb-4">{title}</h3>

      <div className="flex flex-col md:flex-row items-center gap-6">
        {/* Pie Chart */}
        <svg
          viewBox={`0 0 ${size} ${size}`}
          className="w-48 h-48 shrink-0"
        >
          {slices.map((slice, i) => (
            <g key={i}>
              <path
                d={slice.path}
                fill={slice.color}
                stroke="white"
                strokeWidth="2"
              />
              <title>
                {slice.level}: {slice.count} ({slice.percentage}%)
              </title>
            </g>
          ))}
          {/* Center circle for donut effect */}
          <circle
            cx={center}
            cy={center}
            r={radius * 0.6}
            fill="white"
          />
          {/* Total in center */}
          <text
            x={center}
            y={center - 5}
            fontSize="24"
            fontWeight="bold"
            fill="#334155"
            textAnchor="middle"
          >
            {total}
          </text>
          <text
            x={center}
            y={center + 15}
            fontSize="12"
            fill="#94a3b8"
            textAnchor="middle"
          >
            Total
          </text>
        </svg>

        {/* Legend */}
        <div className="flex-1 space-y-2">
          {slices.map((slice, i) => (
            <div key={i} className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: slice.color }}
                />
                <span className="text-sm text-slate-700">{slice.level}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-bold text-slate-900">{slice.count}</span>
                <span className="text-xs text-slate-400">({slice.percentage}%)</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default PieChart;
