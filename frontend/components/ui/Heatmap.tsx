import React from 'react';

interface HeatmapCell {
  day: number;
  hour: number;
  count: number;
}

interface HeatmapProps {
  data: HeatmapCell[];
  title: string;
}

const Heatmap: React.FC<HeatmapProps> = ({ data, title }) => {
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const maxCount = Math.max(...data.map((d) => d.count), 1);

  // Get color intensity based on count
  const getColor = (count: number): string => {
    if (count === 0) return '#f1f5f9'; // slate-100
    const intensity = count / maxCount;
    if (intensity < 0.25) return '#ccfbf1'; // teal-100
    if (intensity < 0.5) return '#5eead4'; // teal-300
    if (intensity < 0.75) return '#14b8a6'; // teal-500
    return '#0d9488'; // teal-600
  };

  // Group by hour ranges for better mobile display
  const hourRanges = [
    { label: '12a', hours: [0, 1, 2] },
    { label: '3a', hours: [3, 4, 5] },
    { label: '6a', hours: [6, 7, 8] },
    { label: '9a', hours: [9, 10, 11] },
    { label: '12p', hours: [12, 13, 14] },
    { label: '3p', hours: [15, 16, 17] },
    { label: '6p', hours: [18, 19, 20] },
    { label: '9p', hours: [21, 22, 23] },
  ];

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

  return (
    <div className="bg-white rounded-xl border border-slate-300 p-6">
      <h3 className="text-sm font-bold text-slate-700 mb-4">{title}</h3>

      {/* Desktop: Full 24-hour heatmap */}
      <div className="hidden lg:block overflow-x-auto">
        <div className="inline-block min-w-full">
          {/* Hour labels */}
          <div className="flex gap-1 mb-2 ml-12">
            {Array.from({ length: 24 }).map((_, hour) => (
              <div key={hour} className="w-6 text-center">
                <span className="text-xs text-slate-400">
                  {hour === 0 ? '12a' : hour < 12 ? `${hour}a` : hour === 12 ? '12p' : `${hour - 12}p`}
                </span>
              </div>
            ))}
          </div>

          {/* Heatmap rows */}
          {days.map((day, dayIndex) => (
            <div key={dayIndex} className="flex items-center gap-1 mb-1">
              <div className="w-10 text-xs font-medium text-slate-600 text-right">
                {day}
              </div>
              {Array.from({ length: 24 }).map((_, hour) => {
                const cell = data.find((d) => d.day === dayIndex && d.hour === hour);
                const count = cell?.count || 0;
                const color = getColor(count);

                return (
                  <div
                    key={hour}
                    className="w-6 h-6 rounded transition-all hover:ring-2 hover:ring-teal-600 cursor-pointer"
                    style={{ backgroundColor: color }}
                    title={`${day} ${hour}:00 - ${count} analyses`}
                  />
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Mobile/Tablet: Simplified 3-hour blocks */}
      <div className="lg:hidden overflow-x-auto">
        <div className="inline-block min-w-full">
          {/* Hour range labels */}
          <div className="flex gap-1 mb-2 ml-12">
            {hourRanges.map((range, i) => (
              <div key={i} className="w-10 text-center">
                <span className="text-xs text-slate-400">{range.label}</span>
              </div>
            ))}
          </div>

          {/* Heatmap rows */}
          {days.map((day, dayIndex) => (
            <div key={dayIndex} className="flex items-center gap-1 mb-1">
              <div className="w-10 text-xs font-medium text-slate-600 text-right">
                {day}
              </div>
              {hourRanges.map((range, rangeIndex) => {
                // Sum counts for hours in this range
                const totalCount = range.hours.reduce((sum, hour) => {
                  const cell = data.find((d) => d.day === dayIndex && d.hour === hour);
                  return sum + (cell?.count || 0);
                }, 0);
                const color = getColor(totalCount);

                return (
                  <div
                    key={rangeIndex}
                    className="w-10 h-10 rounded transition-all hover:ring-2 hover:ring-teal-600 cursor-pointer flex items-center justify-center"
                    style={{ backgroundColor: color }}
                    title={`${day} ${range.label} - ${totalCount} analyses`}
                  >
                    {totalCount > 0 && (
                      <span className="text-xs font-bold text-slate-700">
                        {totalCount}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="mt-4 pt-4 border-t border-slate-100">
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Less activity</span>
          <div className="flex gap-1">
            {[0, 0.25, 0.5, 0.75, 1].map((intensity, i) => (
              <div
                key={i}
                className="w-4 h-4 rounded"
                style={{ backgroundColor: getColor(intensity * maxCount) }}
              />
            ))}
          </div>
          <span>More activity</span>
        </div>
      </div>
    </div>
  );
};

export default Heatmap;
