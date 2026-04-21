import React, { useState, useEffect, useRef } from 'react';
import Button from './ui/Button';
import { supabase } from '../lib/supabase';
import { decryptImage, blobToDataUrl } from '../lib/imageEncryption';
import { useDataCache } from '../lib/dataCache';
import type { AnalysisHistoryItem } from '../lib/types';

// Re-export so HistoryDetailScreen can still import from here
export type { AnalysisHistoryItem };

interface HistoryScreenProps {
  onBack: () => void;
  onViewDetails: (item: AnalysisHistoryItem) => void;
}

const PAGE_SIZE = 20;

// Risk level mappings
const RISK_LEVELS = {
  critical: ['mel'],
  high: ['bcc'],
  moderate: ['akiec'],
  low: ['bkl', 'df', 'nv', 'vasc'],
} as const;

const LESION_TYPES = [
  { id: 'mel', name: 'Melanoma' },
  { id: 'bcc', name: 'Basal Cell Carcinoma' },
  { id: 'akiec', name: 'Actinic Keratoses' },
  { id: 'bkl', name: 'Benign Keratosis' },
  { id: 'df', name: 'Dermatofibroma' },
  { id: 'nv', name: 'Melanocytic Nevi' },
  { id: 'vasc', name: 'Vascular Lesions' },
];

const SORT_OPTIONS = [
  { value: 'created_at_desc', label: 'Newest First' },
  { value: 'created_at_asc', label: 'Oldest First' },
  { value: 'confidence_desc', label: 'Highest Confidence' },
  { value: 'confidence_asc', label: 'Lowest Confidence' },
];

const TIME_PERIOD_OPTIONS = [
  { value: 'all', label: 'All Time' },
  { value: '7d', label: 'Last 7 Days' },
  { value: '30d', label: 'Last Month' },
  { value: '365d', label: 'Last Year' },
];
const FILTER_DROPDOWN_CLASS =
  'w-full appearance-none px-3.5 py-2.5 pr-10 text-sm border border-slate-300 rounded-lg bg-gradient-to-b from-white to-slate-50 text-slate-700 shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500 transition-colors';

interface FilterState {
  timePeriod: string;
  lesionTypes: Set<string>;
  riskLevels: Set<keyof typeof RISK_LEVELS>;
  needsReview: boolean;
  searchText: string;
  sortBy: string;
}

const HistoryScreen: React.FC<HistoryScreenProps> = ({ onBack, onViewDetails }) => {
  const {
    historyCache,
    fetchHistoryCache,
    userId: cachedUserId,
    invalidateDashboard,
  } = useDataCache();
  const [historyItems, setHistoryItems] = useState<AnalysisHistoryItem[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [page, setPage] = useState(0);
  const [userId, setUserId] = useState<string | null>(cachedUserId);
  const [isRefreshing, setIsRefreshing] = useState(false); // subtle loading indicator

  // Selection / deletion state
  const [selectMode, setSelectMode] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirmModal, setConfirmModal] = useState<'selected' | 'all' | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Filter state - auto-apply (no pending state needed)
  const [filters, setFilters] = useState<FilterState>({
    timePeriod: 'all',
    lesionTypes: new Set<string>(),
    riskLevels: new Set<keyof typeof RISK_LEVELS>(),
    needsReview: false,
    searchText: '',
    sortBy: 'created_at_desc',
  });
  const [showTypeDropdown, setShowTypeDropdown] = useState(false);
  const [showRiskDropdown, setShowRiskDropdown] = useState(false);
  const searchDebounceRef = useRef<NodeJS.Timeout | undefined>(undefined);

  const modalRef = useRef<HTMLDivElement>(null);
  const cancelBtnRef = useRef<HTMLButtonElement>(null);
  const initialLoadRef = useRef(true);
  const typeDropdownRef = useRef<HTMLDivElement>(null);
  const riskDropdownRef = useRef<HTMLDivElement>(null);

  // Helper to build filtered query
  const buildQuery = (start: number, end: number, uid?: string) => {
    const targetUserId = uid ?? userId ?? '';
    let query = supabase
      .from('analyses')
      .select(
        'id, created_at, predicted_class_id, predicted_class_name, confidence, image_url, gradcam_image_url, all_scores, notes, metadata'
      )
      .eq('user_id', targetUserId);

    // Time period filter
    if (filters.timePeriod !== 'all') {
      const now = new Date();
      let startDate: Date;

      switch (filters.timePeriod) {
        case '7d':
          startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
          break;
        case '30d':
          startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
          break;
        case '365d':
          startDate = new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000);
          break;
        default:
          startDate = new Date(0); // Beginning of time
      }
      query = query.gte('created_at', startDate.toISOString());
    }

    // Lesion type filter
    if (filters.lesionTypes.size > 0) {
      query = query.in('predicted_class_id', Array.from(filters.lesionTypes));
    }

    // Risk level filter (map to lesion types)
    if (filters.riskLevels.size > 0) {
      const lesionTypesFromRisk = Array.from(filters.riskLevels).flatMap(
        (level) => RISK_LEVELS[level]
      );
      if (lesionTypesFromRisk.length > 0) {
        // If both lesion types AND risk levels are selected, intersect them
        if (filters.lesionTypes.size > 0) {
          const intersection = lesionTypesFromRisk.filter((type) => filters.lesionTypes.has(type));
          if (intersection.length === 0) {
            // No overlap - return empty result
            query = query.in('predicted_class_id', ['__NONE__']);
          } else {
            query = query.in('predicted_class_id', intersection);
          }
        } else {
          query = query.in('predicted_class_id', lesionTypesFromRisk);
        }
      }
    }

    // Needs review filter (critical/high risk types: MEL, BCC, AKIEC)
    if (filters.needsReview) {
      const needsReviewTypes = ['mel', 'bcc', 'akiec'] as const;
      // If lesion types are already filtered, intersect with needs review types
      if (filters.lesionTypes.size > 0) {
        const intersection = needsReviewTypes.filter((type) => filters.lesionTypes.has(type));
        if (intersection.length === 0) {
          query = query.in('predicted_class_id', ['__NONE__']);
        } else {
          query = query.in('predicted_class_id', intersection);
        }
      } else if (filters.riskLevels.size > 0) {
        // If risk levels are filtered, intersect with needs review types
        const lesionTypesFromRisk = Array.from(filters.riskLevels).flatMap(
          (level) => RISK_LEVELS[level]
        );
        const intersection = needsReviewTypes.filter((type) => lesionTypesFromRisk.includes(type));
        if (intersection.length === 0) {
          query = query.in('predicted_class_id', ['__NONE__']);
        } else {
          query = query.in('predicted_class_id', intersection);
        }
      } else {
        query = query.in('predicted_class_id', needsReviewTypes);
      }
    }

    // Notes search filter
    if (filters.searchText.trim()) {
      query = query.ilike('notes', `%${filters.searchText.trim()}%`);
    }

    // Sort - handle field names with underscores (e.g., 'created_at')
    const sortMapping: Record<string, { field: string; ascending: boolean }> = {
      created_at_desc: { field: 'created_at', ascending: false },
      created_at_asc: { field: 'created_at', ascending: true },
      confidence_desc: { field: 'confidence', ascending: false },
      confidence_asc: { field: 'confidence', ascending: true },
    };
    const sortConfig = sortMapping[filters.sortBy] || sortMapping.created_at_desc;
    query = query.order(sortConfig.field, { ascending: sortConfig.ascending });

    // Pagination
    query = query.range(start, end);

    return query;
  };

  const mapRows = async (
    data: Record<string, unknown>[],
    userId: string
  ): Promise<AnalysisHistoryItem[]> => {
    // Collect all image paths (both original and gradcam)
    const imagePaths = data
      .map((row) => row.image_url as string | null)
      .filter((p): p is string => !!p && !p.startsWith('http'));

    const gradcamPaths = data
      .map((row) => row.gradcam_image_url as string | null)
      .filter((p): p is string => !!p && !p.startsWith('http'));

    const allPaths = [...imagePaths, ...gradcamPaths];

    let imageUrlMap: Record<string, string> = {};
    if (allPaths.length > 0) {
      const { data: signed } = await supabase.storage
        .from('analysis-images')
        .createSignedUrls(allPaths, 60 * 60); // 1-hour expiry

      if (signed) {
        // Fetch and decrypt encrypted images (or use signed URLs for old unencrypted images)
        await Promise.all(
          signed.map(async (entry) => {
            const { path, signedUrl } = entry;
            if (!path || !signedUrl) return;
            try {
              // Check if the file is encrypted (has .enc extension)
              const isEncrypted = path.endsWith('.enc');

              if (isEncrypted) {
                // New encrypted images: fetch and decrypt
                const response = await fetch(signedUrl);
                if (!response.ok) return;
                const encryptedBlob = await response.blob();

                // Decrypt the image
                const decryptedBlob = await decryptImage(encryptedBlob, userId);

                // Convert to data URL for display
                const dataUrl = await blobToDataUrl(decryptedBlob);
                imageUrlMap[path] = dataUrl;
              } else {
                // Old unencrypted images: use signed URL directly
                imageUrlMap[path] = signedUrl;
              }
            } catch (err) {
              console.error(`Failed to process image ${path}:`, err);
              // Silently skip - image won't display but won't break the UI
            }
          })
        );
      }
    }

    return data.map((row) => {
      const rawUrl = row.image_url as string | null;
      const rawGradcamUrl = row.gradcam_image_url as string | null;

      const imageUrl =
        rawUrl && !rawUrl.startsWith('http') ? imageUrlMap[rawUrl] : (rawUrl ?? undefined);
      const gradcamUrl =
        rawGradcamUrl && !rawGradcamUrl.startsWith('http')
          ? imageUrlMap[rawGradcamUrl]
          : (rawGradcamUrl ?? undefined);

      return {
        id: row.id as string,
        createdAt: row.created_at as string,
        date: new Date(row.created_at as string).toLocaleDateString('en-US', {
          year: 'numeric',
          month: 'short',
          day: 'numeric',
        }),
        time: new Date(row.created_at as string).toLocaleTimeString('en-US', {
          hour: '2-digit',
          minute: '2-digit',
        }),
        classId: row.predicted_class_id as string,
        className: row.predicted_class_name as string,
        confidence: row.confidence as number,
        imageUrl,
        imagePath: rawUrl && !rawUrl.startsWith('http') ? rawUrl : undefined,
        gradcamUrl,
        gradcamPath: rawGradcamUrl && !rawGradcamUrl.startsWith('http') ? rawGradcamUrl : undefined,
        allScores: (row.all_scores as AnalysisHistoryItem['allScores']) ?? undefined,
        notes: (row.notes as string | null) ?? undefined,
        metadata: (() => {
          const m = row.metadata as Record<string, unknown> | null;
          if (!m) return null;
          return {
            ageApprox: (m.age_approx as number | null) ?? null,
            sex: (m.sex as string | null) ?? null,
            anatomSite: (m.anatom_site as string | null) ?? null,
          };
        })(),
      };
    });
  };

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        // Use cached userId if available
        if (!userId && cachedUserId) {
          setUserId(cachedUserId);
        }

        const uid = userId || cachedUserId;

        // Check if we can use cached data (no filters applied)
        const hasNoFilters =
          filters.timePeriod === 'all' &&
          filters.lesionTypes.size === 0 &&
          filters.riskLevels.size === 0 &&
          !filters.needsReview &&
          !filters.searchText.trim() &&
          filters.sortBy === 'created_at_desc';

        if (hasNoFilters && historyCache.data && !historyCache.error) {
          // Use cached data
          setHistoryItems(historyCache.data);
          setHasMore(historyCache.data.length === PAGE_SIZE);
          setLoadingHistory(false);
          return;
        }

        // Otherwise, fetch from cache (which will fetch from Supabase if needed)
        if (hasNoFilters) {
          const cachedData = await fetchHistoryCache();
          if (cachedData) {
            setHistoryItems(cachedData);
            setHasMore(cachedData.length === PAGE_SIZE);
            setLoadingHistory(false);
            return;
          }
        }

        // If we have filters or cache failed, fetch directly
        const {
          data: { user },
        } = await supabase.auth.getUser();
        const fetchUid = user?.id ?? null;
        if (fetchUid && !uid) setUserId(fetchUid);

        const { data, error } = await buildQuery(0, PAGE_SIZE - 1, fetchUid ?? undefined);

        if (error) throw error;
        const rows = data ?? [];
        setHistoryItems(await mapRows(rows, fetchUid ?? ''));
        setHasMore(rows.length === PAGE_SIZE);
      } catch {
        setHistoryError('Could not load analysis history. Please try again.');
      } finally {
        setLoadingHistory(false);
      }
    };
    fetchHistory();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Refetch when filters change (skip initial load)
  useEffect(() => {
    // Skip on initial load - the first useEffect handles that
    if (initialLoadRef.current) {
      initialLoadRef.current = false;
      return;
    }

    if (!userId) return;

    const refetch = async () => {
      setIsRefreshing(true);
      setHistoryError(null);
      setPage(0);
      try {
        const { data, error } = await buildQuery(0, PAGE_SIZE - 1);
        if (error) throw error;
        const rows = data ?? [];
        setHistoryItems(await mapRows(rows, userId));
        setHasMore(rows.length === PAGE_SIZE);
      } catch {
        setHistoryError('Could not load analysis history. Please try again.');
        setHistoryItems([]);
      } finally {
        setIsRefreshing(false);
      }
    };

    refetch();
  }, [filters, userId]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleLoadMore = async () => {
    if (!userId) return;
    setLoadingMore(true);
    const nextPage = page + 1;
    try {
      const { data, error } = await buildQuery(
        nextPage * PAGE_SIZE,
        (nextPage + 1) * PAGE_SIZE - 1
      );

      if (error) throw error;
      const rows = data ?? [];
      const mappedRows = await mapRows(rows, userId);
      setHistoryItems((prev) => [...prev, ...mappedRows]);
      setHasMore(rows.length === PAGE_SIZE);
      setPage(nextPage);
    } catch {
      setHistoryError('Could not load more records. Please try again.');
    } finally {
      setLoadingMore(false);
    }
  };

  // ── Selection helpers ──────────────────────────────────────────────────────
  const toggleSelectMode = () => {
    setSelectMode((prev) => !prev);
    setSelected(new Set());
  };

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const allSelected = historyItems.length > 0 && historyItems.every((i) => selected.has(i.id));

  const toggleSelectAll = () => {
    setSelected(allSelected ? new Set() : new Set(historyItems.map((i) => i.id)));
  };

  // ── Delete handlers ────────────────────────────────────────────────────────
  const handleDeleteSelected = async () => {
    setDeleting(true);
    setHistoryError(null);
    const ids = [...selected];

    try {
      const paths = ids.flatMap((id) => {
        const item = historyItems.find((i) => i.id === id);
        const itemPaths: string[] = [];
        if (item?.imagePath) itemPaths.push(item.imagePath);
        if (item?.gradcamPath) itemPaths.push(item.gradcamPath);
        return itemPaths;
      });

      if (paths.length > 0) {
        const { error: storageError } = await supabase.storage
          .from('analysis-images')
          .remove(paths);
        if (storageError) {
          throw new Error(`Failed to remove images: ${storageError.message}`);
        }
      }

      const { error: dbError } = await supabase.from('analyses').delete().in('id', ids);
      if (dbError) {
        throw new Error(`Failed to delete records: ${dbError.message}`);
      }

      setHistoryItems((prev) => prev.filter((i) => !ids.includes(i.id)));
      setSelected(new Set());
      setSelectMode(false);
      setConfirmModal(null);
      // Invalidate cache so dashboard stats get refreshed
      invalidateDashboard();
    } catch (err) {
      setHistoryError(
        err instanceof Error ? err.message : 'Could not delete selected records. Please try again.'
      );
    } finally {
      setDeleting(false);
    }
  };

  const handleClearAll = async () => {
    setDeleting(true);
    setHistoryError(null);

    try {
      const {
        data: { user },
        error: userError,
      } = await supabase.auth.getUser();
      if (userError || !user) throw new Error('Not authenticated');

      const { data: allRows, error: fetchError } = await supabase
        .from('analyses')
        .select('image_url, gradcam_image_url')
        .eq('user_id', user.id)
        .or('image_url.not.is.null,gradcam_image_url.not.is.null');
      if (fetchError) {
        throw new Error(`Failed to fetch records: ${fetchError.message}`);
      }

      const paths = (allRows ?? []).flatMap((r) => {
        const rowPaths: string[] = [];
        const imageUrl = r.image_url as string | null;
        const gradcamUrl = r.gradcam_image_url as string | null;
        if (imageUrl && !imageUrl.startsWith('http')) rowPaths.push(imageUrl);
        if (gradcamUrl && !gradcamUrl.startsWith('http')) rowPaths.push(gradcamUrl);
        return rowPaths;
      });
      if (paths.length > 0) {
        const { error: storageError } = await supabase.storage
          .from('analysis-images')
          .remove(paths);
        if (storageError) {
          throw new Error(`Failed to remove images: ${storageError.message}`);
        }
      }

      const { error: dbError } = await supabase.from('analyses').delete().eq('user_id', user.id);
      if (dbError) {
        throw new Error(`Failed to clear history: ${dbError.message}`);
      }

      setHistoryItems([]);
      setSelected(new Set());
      setSelectMode(false);
      setHasMore(false);
      setConfirmModal(null);
      // Invalidate cache so dashboard stats get refreshed
      invalidateDashboard();
    } catch (err) {
      setHistoryError(
        err instanceof Error ? err.message : 'Could not clear history. Please try again.'
      );
    } finally {
      setDeleting(false);
    }
  };

  const hasItems = historyItems.length > 0;

  // Filter helpers - auto-apply
  const toggleLesionType = (typeId: string) => {
    setFilters((prev) => {
      const newTypes = new Set(prev.lesionTypes);
      if (newTypes.has(typeId)) {
        newTypes.delete(typeId);
      } else {
        newTypes.add(typeId);
      }
      return { ...prev, lesionTypes: newTypes };
    });
  };

  const toggleRiskLevel = (level: keyof typeof RISK_LEVELS) => {
    setFilters((prev) => {
      const newLevels = new Set(prev.riskLevels);
      if (newLevels.has(level)) {
        newLevels.delete(level);
      } else {
        newLevels.add(level);
      }
      return { ...prev, riskLevels: newLevels };
    });
  };

  const clearFilters = () => {
    setFilters({
      timePeriod: 'all',
      lesionTypes: new Set<string>(),
      riskLevels: new Set<keyof typeof RISK_LEVELS>(),
      needsReview: false,
      searchText: '',
      sortBy: 'created_at_desc',
    });
  };

  // Debounced search
  const handleSearchChange = (value: string) => {
    if (searchDebounceRef.current) {
      clearTimeout(searchDebounceRef.current);
    }

    searchDebounceRef.current = setTimeout(() => {
      setFilters((prev) => ({ ...prev, searchText: value }));
    }, 500);
  };

  const activeFilterCount =
    (filters.timePeriod !== 'all' ? 1 : 0) +
    filters.lesionTypes.size +
    filters.riskLevels.size +
    (filters.needsReview ? 1 : 0) +
    (filters.searchText.trim() ? 1 : 0);

  const hasActiveFilters = activeFilterCount > 0;

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (typeDropdownRef.current && !typeDropdownRef.current.contains(event.target as Node)) {
        setShowTypeDropdown(false);
      }
      if (riskDropdownRef.current && !riskDropdownRef.current.contains(event.target as Node)) {
        setShowRiskDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Modal: Escape-to-close, focus trapping, and initial focus
  useEffect(() => {
    if (!confirmModal) return;

    cancelBtnRef.current?.focus();

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !deleting) {
        setConfirmModal(null);
        return;
      }
      if (e.key === 'Tab' && modalRef.current) {
        const focusable = modalRef.current.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [confirmModal, deleting]);

  return (
    <div className="flex-1 flex flex-col bg-slate-50 text-slate-900 pb-12">
      <main className="max-w-4xl mx-auto w-full px-4 sm:px-6 py-8 flex flex-col gap-8">
        {/* Header */}
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <button
              onClick={onBack}
              className="p-2 hover:bg-slate-100 rounded-full transition-colors text-slate-400 hover:text-slate-600"
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
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Past Analyses</h1>
          </div>

          {/* Action buttons */}
          {!loadingHistory && hasItems && (
            <div className="flex items-center gap-2">
              <button
                onClick={toggleSelectMode}
                className={[
                  'text-xs font-bold px-3 py-1.5 rounded-lg border transition-colors',
                  selectMode
                    ? 'bg-slate-800 text-white border-slate-800'
                    : 'bg-white text-slate-600 border-slate-300 hover:border-slate-300 hover:text-slate-800',
                ].join(' ')}
              >
                {selectMode ? 'Cancel' : 'Select'}
              </button>
              {!selectMode && (
                <button
                  onClick={() => setConfirmModal('all')}
                  className="text-xs font-bold px-3 py-1.5 rounded-lg border border-red-200 text-red-500 bg-white hover:bg-red-50 hover:border-red-300 transition-colors"
                >
                  Clear History
                </button>
              )}
            </div>
          )}
        </div>

        {/* ──Compact Inline Filter Bar ── */}
        {!loadingHistory && (hasItems || hasActiveFilters) && (
          <div className="bg-white rounded-2xl border border-slate-400 shadow-sm p-4">
            <div className="flex flex-col gap-3">
              {/* First Row: Search + Time Period */}
              <div className="flex flex-col sm:flex-row gap-3">
                {/* Search Bar */}
                <div className="flex-1 relative">
                  <input
                    type="text"
                    placeholder="Search notes..."
                    defaultValue={filters.searchText}
                    onChange={(e) => handleSearchChange(e.target.value)}
                    className="w-full px-4 py-2.5 pl-10 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent"
                  />
                  <svg
                    className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                  </svg>
                </div>

                {/* Time Period Dropdown */}
                <div className="sm:w-44 relative">
                  <select
                    value={filters.timePeriod}
                    onChange={(e) => setFilters({ ...filters, timePeriod: e.target.value })}
                    className={FILTER_DROPDOWN_CLASS}
                  >
                    {TIME_PERIOD_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                  <div className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-slate-400">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 9l-7 7-7-7"
                      />
                    </svg>
                  </div>
                </div>
              </div>

              {/* Second Row: Type, Risk, Needs Review, Sort */}
              <div className="flex flex-wrap gap-2">
                {/* Type Multi-Select Dropdown */}
                <div className="relative" ref={typeDropdownRef}>
                  <button
                    onClick={() => setShowTypeDropdown(!showTypeDropdown)}
                    className={[
                      'px-3 py-2 text-sm font-medium border rounded-lg transition-colors flex items-center gap-2',
                      filters.lesionTypes.size > 0
                        ? 'border-teal-300 bg-teal-50 text-teal-700 hover:bg-teal-100'
                        : 'border-slate-300 bg-white text-slate-600 hover:border-slate-400',
                    ].join(' ')}
                  >
                    Type
                    {filters.lesionTypes.size > 0 && (
                      <span className="px-1.5 py-0.5 text-xs font-bold rounded bg-teal-600 text-white">
                        {filters.lesionTypes.size}
                      </span>
                    )}
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 9l-7 7-7-7"
                      />
                    </svg>
                  </button>

                  {showTypeDropdown && (
                    <div className="absolute z-50 mt-2 w-64 bg-white border border-slate-300 rounded-lg shadow-xl py-2">
                      {LESION_TYPES.map((type) => (
                        <label
                          key={type.id}
                          className="flex items-center gap-2 px-3 py-2 hover:bg-slate-50 cursor-pointer"
                        >
                          <input
                            type="checkbox"
                            checked={filters.lesionTypes.has(type.id)}
                            onChange={() => toggleLesionType(type.id)}
                            className="w-4 h-4 rounded border-slate-300 text-teal-600 accent-teal-600 cursor-pointer"
                          />
                          <span className="text-sm text-slate-700 flex-1">{type.name}</span>
                          <span className="text-xs text-slate-400 uppercase font-mono">
                            {type.id}
                          </span>
                        </label>
                      ))}
                    </div>
                  )}
                </div>

                {/* Risk Level Multi-Select Dropdown */}
                <div className="relative" ref={riskDropdownRef}>
                  <button
                    onClick={() => setShowRiskDropdown(!showRiskDropdown)}
                    className={[
                      'px-3 py-2 text-sm font-medium border rounded-lg transition-colors flex items-center gap-2',
                      filters.riskLevels.size > 0
                        ? 'border-teal-300 bg-teal-50 text-teal-700 hover:bg-teal-100'
                        : 'border-slate-300 bg-white text-slate-600 hover:border-slate-300',
                    ].join(' ')}
                  >
                    Risk Level
                    {filters.riskLevels.size > 0 && (
                      <span className="px-1.5 py-0.5 text-xs font-bold rounded bg-teal-600 text-white">
                        {filters.riskLevels.size}
                      </span>
                    )}
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 9l-7 7-7-7"
                      />
                    </svg>
                  </button>

                  {showRiskDropdown && (
                    <div className="absolute z-50 mt-2 w-56 bg-white border border-slate-200 rounded-lg shadow-xl py-2">
                      {(Object.keys(RISK_LEVELS) as Array<keyof typeof RISK_LEVELS>).map(
                        (level) => (
                          <label
                            key={level}
                            className="flex items-center gap-2 px-3 py-2 hover:bg-slate-50 cursor-pointer"
                          >
                            <input
                              type="checkbox"
                              checked={filters.riskLevels.has(level)}
                              onChange={() => toggleRiskLevel(level)}
                              className="w-4 h-4 rounded border-slate-300 text-teal-600 accent-teal-600 cursor-pointer"
                            />
                            <span
                              className={[
                                'text-sm font-medium capitalize',
                                level === 'critical'
                                  ? 'text-red-600'
                                  : level === 'high'
                                    ? 'text-orange-600'
                                    : level === 'moderate'
                                      ? 'text-amber-600'
                                      : 'text-emerald-600',
                              ].join(' ')}
                            >
                              {level}
                            </span>
                          </label>
                        )
                      )}
                    </div>
                  )}
                </div>

                {/* Needs Review Toggle */}
                <button
                  onClick={() => setFilters({ ...filters, needsReview: !filters.needsReview })}
                  className={[
                    'px-3 py-2 text-sm font-medium border rounded-lg transition-colors flex items-center gap-2',
                    filters.needsReview
                      ? 'border-amber-300 bg-amber-50 text-amber-700 hover:bg-amber-100'
                      : 'border-slate-300 bg-white text-slate-600 hover:border-slate-300',
                  ].join(' ')}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                    />
                  </svg>
                  Needs Review
                </button>

                {/* Sort Dropdown */}
                <div className="sm:w-48 ml-auto relative">
                  <select
                    value={filters.sortBy}
                    onChange={(e) => setFilters({ ...filters, sortBy: e.target.value })}
                    className={FILTER_DROPDOWN_CLASS}
                  >
                    {SORT_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                  <div className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-slate-400">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 9l-7 7-7-7"
                      />
                    </svg>
                  </div>
                </div>

                {/* Clear Filters Button (only when active) */}
                {hasActiveFilters && (
                  <button
                    onClick={clearFilters}
                    className="px-3 py-2 text-sm font-semibold text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-colors"
                  >
                    Clear all
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        <section className="bg-white rounded-3xl border border-slate-300 overflow-hidden shadow-sm">
          {loadingHistory ? (
            <div className="flex items-center justify-center py-16">
              <svg className="animate-spin h-6 w-6 text-teal-600" viewBox="0 0 24 24" fill="none">
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
          ) : historyError ? (
            <div className="py-12 px-6 text-center">
              <p className="text-sm text-red-500 font-medium">{historyError}</p>
            </div>
          ) : historyItems.length === 0 ? (
            <div className="py-16 px-6 text-center">
              {hasActiveFilters ? (
                <>
                  <svg
                    className="w-12 h-12 text-slate-300 mx-auto mb-3"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                  </svg>
                  <p className="text-sm font-medium text-slate-600 mb-1">
                    No analyses match your filters
                  </p>
                  <button
                    onClick={clearFilters}
                    className="text-xs font-semibold text-teal-600 hover:text-teal-700 mt-2 transition-colors"
                  >
                    Clear filters to see all analyses
                  </button>
                </>
              ) : (
                <>
                  <p className="text-sm text-slate-400">No analyses found.</p>
                  <p className="text-xs text-slate-300 mt-1">
                    Run your first classification to see it here.
                  </p>
                </>
              )}
            </div>
          ) : (
            <>
              {/* Selection toolbar */}
              {selectMode && (
                <div className="flex items-center justify-between px-4 sm:px-6 py-3 border-b border-slate-200 bg-slate-50">
                  <label className="flex items-center gap-2.5 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={allSelected}
                      onChange={toggleSelectAll}
                      className="w-4 h-4 rounded border-slate-300 text-teal-600 accent-teal-600 cursor-pointer"
                    />
                    <span className="text-xs font-semibold text-slate-600">
                      {allSelected ? 'Deselect all' : 'Select all'}
                    </span>
                  </label>
                  <button
                    disabled={selected.size === 0}
                    onClick={() => setConfirmModal('selected')}
                    className="text-xs font-bold px-3 py-1.5 rounded-lg bg-red-500 text-white hover:bg-red-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    Delete {selected.size > 0 ? `${selected.size} selected` : 'selected'}
                  </button>
                </div>
              )}

              {/* Mobile card layout (< md) */}
              <ul className="divide-y divide-slate-100 md:hidden">
                {historyItems.map((item) => (
                  <li
                    key={item.id}
                    className={[
                      'flex items-center gap-4 px-4 py-4 transition-all',
                      selectMode && selected.has(item.id) ? 'bg-red-50/60' : '',
                    ].join(' ')}
                  >
                    {selectMode && (
                      <input
                        type="checkbox"
                        checked={selected.has(item.id)}
                        onChange={() => toggleSelect(item.id)}
                        aria-label={`Select case DRM-${item.id.slice(0, 8).toUpperCase()}`}
                        className="w-4 h-4 rounded border-slate-300 accent-teal-600 cursor-pointer shrink-0"
                      />
                    )}
                    <div className="w-12 h-12 rounded-lg bg-slate-100 flex items-center justify-center text-slate-300 border border-slate-200 overflow-hidden shrink-0">
                      {item.imageUrl ? (
                        <img
                          src={item.imageUrl}
                          alt="Lesion thumbnail"
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
                            strokeWidth={1}
                            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                          />
                        </svg>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-0.5">
                        <span className="text-sm font-bold text-teal-700 uppercase">
                          {item.classId}
                        </span>
                        <span className="text-[10px] font-bold text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded tracking-tighter">
                          {item.confidence.toFixed(1)}%
                        </span>
                      </div>
                      <p className="text-xs text-slate-500 truncate">{item.className}</p>
                      <p className="text-xs text-slate-400 mt-0.5">
                        {item.date} · {item.time}
                      </p>
                    </div>
                    {!selectMode && (
                      <button
                        onClick={() => onViewDetails(item)}
                        className="shrink-0 text-teal-600 hover:text-teal-700 p-2 rounded-full hover:bg-teal-50 transition-colors"
                        title="View Details"
                      >
                        <svg
                          className="w-4 h-4"
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
                    )}
                  </li>
                ))}
              </ul>

              {/* Desktop table layout (md+) */}
              <div className="hidden md:block overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-200">
                      {selectMode && <th className="pl-6 py-4 w-10" />}
                      <th className="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest">
                        Lesion Context
                      </th>
                      <th className="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest">
                        Date / Time
                      </th>
                      <th className="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest">
                        Predicted Class
                      </th>
                      <th className="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest text-right">
                        Details
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {historyItems.map((item) => (
                      <tr
                        key={item.id}
                        className={[
                          'hover:bg-slate-50/50 transition-all group',
                          selectMode && selected.has(item.id) ? 'bg-red-50/60' : '',
                        ].join(' ')}
                      >
                        {selectMode && (
                          <td className="pl-6 py-4 w-10">
                            <input
                              type="checkbox"
                              checked={selected.has(item.id)}
                              onChange={() => toggleSelect(item.id)}
                              aria-label={`Select case DRM-${item.id.slice(0, 8).toUpperCase()}`}
                              className="w-4 h-4 rounded border-slate-300 accent-teal-600 cursor-pointer"
                            />
                          </td>
                        )}
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-4">
                            <div className="w-12 h-12 rounded-lg bg-slate-100 flex items-center justify-center text-slate-300 border border-slate-200 overflow-hidden relative">
                              {item.imageUrl ? (
                                <img
                                  src={item.imageUrl}
                                  alt="Lesion thumbnail"
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
                                    strokeWidth={1}
                                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                                  />
                                </svg>
                              )}
                            </div>
                            <div>
                              <div className="text-xs font-bold text-slate-400 uppercase tracking-tighter">
                                Case Reference
                              </div>
                              <div className="text-sm font-semibold text-slate-700 font-mono">
                                DRM-{item.id.slice(0, 8).toUpperCase()}
                              </div>
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
                              <span className="text-sm font-bold text-teal-700 uppercase">
                                {item.classId}
                              </span>
                              <span className="text-[10px] font-bold text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded tracking-tighter">
                                {item.confidence.toFixed(1)}%
                              </span>
                            </div>
                            <span className="text-xs text-slate-500">{item.className}</span>
                          </div>
                        </td>
                        <td className="px-6 py-4 text-right">
                          {!selectMode && (
                            <button
                              onClick={() => onViewDetails(item)}
                              className="text-teal-600 hover:text-teal-700 text-sm font-bold uppercase tracking-widest flex items-center gap-1 justify-end ml-auto group-hover:translate-x-1 transition-transform"
                            >
                              View Details
                              <svg
                                className="w-4 h-4"
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
                          )}
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
                        Loading…
                      </>
                    ) : (
                      <>
                        Load more records
                        <svg
                          className="w-4 h-4"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M19 9l-7 7-7-7"
                          />
                        </svg>
                      </>
                    )}
                  </button>
                </div>
              )}
            </>
          )}
        </section>

        <div className="p-6 bg-blue-50/30 rounded-2xl border border-blue-100">
          <div className="flex gap-3">
            <div className="text-blue-500">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
            <p className="text-xs text-blue-700 font-medium leading-relaxed">
              Analysis history is fetched from your account, ordered newest first. Records are
              persisted across devices and sessions.
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
          Clinical Support Tool. Designed to assist medical professionals.
        </p>
      </footer>

      {/* ── Confirmation modal ─────────────────────────────────────────────── */}
      {confirmModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/40 backdrop-blur-sm"
            onClick={() => !deleting && setConfirmModal(null)}
          />
          {/* Dialog */}
          <div
            ref={modalRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-confirm-title"
            aria-describedby="delete-confirm-desc"
            className="relative bg-white rounded-2xl shadow-xl max-w-sm w-full p-6 flex flex-col gap-5"
          >
            {/* Icon */}
            <div className="flex items-center justify-center w-12 h-12 rounded-full bg-red-50 mx-auto">
              <svg
                className="w-6 h-6 text-red-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
            </div>
            {/* Text */}
            <div className="text-center">
              <h2 id="delete-confirm-title" className="text-base font-bold text-slate-900 mb-1">
                {confirmModal === 'all'
                  ? 'Clear all history?'
                  : `Delete ${selected.size} record${selected.size !== 1 ? 's' : ''}?`}
              </h2>
              <p id="delete-confirm-desc" className="text-sm text-slate-500 leading-relaxed">
                {confirmModal === 'all'
                  ? 'This will permanently delete all your analysis records. This action cannot be undone.'
                  : `This will permanently delete the selected record${selected.size !== 1 ? 's' : ''}. This action cannot be undone.`}
              </p>
            </div>
            {/* Actions */}
            <div className="flex gap-3">
              <button
                ref={cancelBtnRef}
                onClick={() => setConfirmModal(null)}
                disabled={deleting}
                className="flex-1 py-2.5 rounded-xl border border-slate-200 text-sm font-semibold text-slate-600 hover:bg-slate-50 disabled:opacity-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmModal === 'all' ? handleClearAll : handleDeleteSelected}
                disabled={deleting}
                className="flex-1 py-2.5 rounded-xl bg-red-500 text-sm font-bold text-white hover:bg-red-600 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
              >
                {deleting ? (
                  <>
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
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
                    Deleting…
                  </>
                ) : (
                  'Delete permanently'
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HistoryScreen;
