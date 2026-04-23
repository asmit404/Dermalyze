import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { supabase } from './supabase';
import { decryptImage, blobToDataUrl } from './imageEncryption';
import type { AnalysisHistoryItem, DashStats } from './types';
import { SHORT_CLASS_NAMES } from './types';

// Cache TTL in milliseconds (5 minutes)
const CACHE_TTL = 5 * 60 * 1000;

interface CachedData<T> {
  data: T | null;
  timestamp: number | null;
  loading: boolean;
  error: boolean;
}

interface DataCacheContextType {
  // Dashboard cache
  dashboardStats: CachedData<DashStats>;
  fetchDashboardStats: (forceRefresh?: boolean) => Promise<void>;

  // History cache (first page only)
  historyCache: CachedData<AnalysisHistoryItem[]>;
  fetchHistoryCache: (forceRefresh?: boolean) => Promise<AnalysisHistoryItem[] | null>;

  // Cache invalidation
  invalidateAll: () => void;
  invalidateDashboard: () => void;
  invalidateHistory: () => void;

  // User info
  userId: string | null;
  userName: string | null;
}

const DataCacheContext = createContext<DataCacheContextType | undefined>(undefined);

const PAGE_SIZE = 20;

export const DataCacheProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [userId, setUserId] = useState<string | null>(null);
  const [userName, setUserName] = useState<string | null>(null);

  const [dashboardStats, setDashboardStats] = useState<CachedData<DashStats>>({
    data: null,
    timestamp: null,
    loading: false,
    error: false,
  });

  const [historyCache, setHistoryCache] = useState<CachedData<AnalysisHistoryItem[]>>({
    data: null,
    timestamp: null,
    loading: false,
    error: false,
  });

  // Initialize user info on mount
  useEffect(() => {
    const initUser = async () => {
      try {
        const {
          data: { user },
        } = await supabase.auth.getUser();
        if (user) {
          setUserId(user.id);
          setUserName(user.user_metadata?.full_name?.split(' ')[0] ?? '');
        }
      } catch (err) {
        console.error('Failed to get user:', err);
      }
    };
    initUser();

    // Invalidate caches on auth state changes (login/logout) to prevent
    // serving stale data from a previous session.
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event) => {
      if (event === 'SIGNED_IN' || event === 'SIGNED_OUT') {
        setDashboardStats({ data: null, timestamp: null, loading: false, error: false });
        setHistoryCache({ data: null, timestamp: null, loading: false, error: false });
      }
      if (event === 'SIGNED_IN') {
        supabase.auth.getUser().then(({ data: { user } }) => {
          if (user) {
            setUserId(user.id);
            setUserName(user.user_metadata?.full_name?.split(' ')[0] ?? '');
          }
        });
      }
      if (event === 'SIGNED_OUT') {
        setUserId(null);
        setUserName(null);
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  // Check if cache is stale
  const isCacheStale = useCallback((timestamp: number | null): boolean => {
    if (!timestamp) return true;
    return Date.now() - timestamp > CACHE_TTL;
  }, []);

  // Fetch dashboard stats
  const fetchDashboardStats = useCallback(
    async (forceRefresh = false) => {
      // Return cached data if valid and not forcing refresh
      if (!forceRefresh && dashboardStats.data && !isCacheStale(dashboardStats.timestamp)) {
        return;
      }

      setDashboardStats((prev) => ({ ...prev, loading: true, error: false }));

      try {
        const {
          data: { user },
        } = await supabase.auth.getUser();
        if (!user) throw new Error('No user found');

        const [{ data: rpcData, error: rpcErr }, { data: lastRows, error: lastErr }] =
          await Promise.all([
            supabase.rpc('get_dashboard_stats'),
            supabase
              .from('analyses')
              .select(
                'predicted_class_id, predicted_class_name, confidence, created_at, image_url, gradcam_image_url'
              )
              .eq('user_id', user.id)
              .order('created_at', { ascending: false })
              .limit(1),
          ]);

        if (rpcErr || !rpcData) throw new Error('Failed to fetch dashboard stats');

        interface RpcResult {
          total: number;
          this_month: number;
          avg_confidence: number | null;
          needs_review: number;
          class_counts: { id: string; name: string; count: number }[] | null;
        }

        const rpc = rpcData as RpcResult;
        const last = !lastErr && lastRows && lastRows.length > 0 ? lastRows[0] : null;

        let resolvedImageUrl: string | null = last?.image_url ?? null;
        if (resolvedImageUrl && !resolvedImageUrl.startsWith('http')) {
          try {
            const isEncrypted = resolvedImageUrl.endsWith('.enc');
            const { data: signed } = await supabase.storage
              .from('analysis-images')
              .createSignedUrl(resolvedImageUrl, 60 * 60);

            if (signed?.signedUrl) {
              if (isEncrypted) {
                const response = await fetch(signed.signedUrl);
                if (response.ok) {
                  const encryptedBlob = await response.blob();
                  const decryptedBlob = await decryptImage(encryptedBlob, user.id);
                  resolvedImageUrl = await blobToDataUrl(decryptedBlob);
                } else {
                  resolvedImageUrl = null;
                }
              } else {
                resolvedImageUrl = signed.signedUrl;
              }
            } else {
              resolvedImageUrl = null;
            }
          } catch (err) {
            console.error('Failed to process dashboard image:', err);
            resolvedImageUrl = null;
          }
        }

        const lastAnalysis = last
          ? {
              className: SHORT_CLASS_NAMES[last.predicted_class_id] ?? last.predicted_class_name,
              classId: last.predicted_class_id,
              confidence: last.confidence,
              date: new Date(last.created_at).toLocaleDateString('en-GB', {
                day: '2-digit',
                month: 'short',
                year: 'numeric',
              }),
              imageUrl: resolvedImageUrl,
            }
          : null;

        const classCounts = (rpc.class_counts ?? []).map((c) => ({
          id: c.id,
          name: SHORT_CLASS_NAMES[c.id] ?? c.name,
          count: c.count,
        }));

        const stats: DashStats = {
          total: rpc.total,
          thisMonth: rpc.this_month,
          avgConfidence: rpc.avg_confidence,
          needsReview: rpc.needs_review,
          classCounts,
          lastAnalysis,
        };

        setDashboardStats({
          data: stats,
          timestamp: Date.now(),
          loading: false,
          error: false,
        });
      } catch (err) {
        console.error('Failed to fetch dashboard stats:', err);
        setDashboardStats((prev) => ({ ...prev, loading: false, error: true }));
      }
    },
    [dashboardStats.data, dashboardStats.timestamp, isCacheStale]
  );

  // Fetch history cache (first page only)
  const fetchHistoryCache = useCallback(
    async (forceRefresh = false): Promise<AnalysisHistoryItem[] | null> => {
      // Return cached data if valid and not forcing refresh
      if (!forceRefresh && historyCache.data && !isCacheStale(historyCache.timestamp)) {
        return historyCache.data;
      }

      if (!userId) return null;

      setHistoryCache((prev) => ({ ...prev, loading: true, error: false }));

      try {
        const { data, error } = await supabase
          .from('analyses')
          .select(
            'id, created_at, predicted_class_id, predicted_class_name, confidence, image_url, gradcam_image_url, all_scores, notes, trust_recommendation, trust_uncertainty_score, trust_quality_flags'
          )
          .eq('user_id', userId)
          .order('created_at', { ascending: false })
          .range(0, PAGE_SIZE - 1);

        if (error) throw error;

        const rows = data ?? [];

        // Map rows with image processing (both original and gradcam)
        const imagePaths = rows
          .map((row) => row.image_url as string | null)
          .filter((p): p is string => !!p && !p.startsWith('http'));

        const gradcamPaths = rows
          .map((row) => row.gradcam_image_url as string | null)
          .filter((p): p is string => !!p && !p.startsWith('http'));

        const allPaths = [...imagePaths, ...gradcamPaths];

        let imageUrlMap: Record<string, string> = {};
        const failedPaths = new Set<string>();
        if (allPaths.length > 0) {
          const { data: signed } = await supabase.storage
            .from('analysis-images')
            .createSignedUrls(allPaths, 60 * 60);

          if (signed) {
            await Promise.all(
              signed.map(async (entry) => {
                const { path, signedUrl } = entry;
                if (!path || !signedUrl) return;
                try {
                  const isEncrypted = path.endsWith('.enc');

                  if (isEncrypted) {
                    const response = await fetch(signedUrl);
                    if (!response.ok) return;
                    const encryptedBlob = await response.blob();
                    const decryptedBlob = await decryptImage(encryptedBlob, userId);
                    const dataUrl = await blobToDataUrl(decryptedBlob);
                    imageUrlMap[path] = dataUrl;
                  } else {
                    imageUrlMap[path] = signedUrl;
                  }
                } catch (err) {
                  console.error(`Failed to process image ${path}:`, err);
                  failedPaths.add(path);
                }
              })
            );
          }
        }

        const mappedRows: AnalysisHistoryItem[] = rows.map((row) => {
          const rawUrl = row.image_url as string | null;
          const rawGradcamUrl = row.gradcam_image_url as string | null;

          const imageUrl =
            rawUrl && !rawUrl.startsWith('http') ? imageUrlMap[rawUrl] : (rawUrl ?? undefined);
          const gradcamUrl =
            rawGradcamUrl && !rawGradcamUrl.startsWith('http')
              ? imageUrlMap[rawGradcamUrl]
              : (rawGradcamUrl ?? undefined);

          return {
            id: row.id,
            createdAt: row.created_at,
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
            imageUrl,
            imagePath: rawUrl && !rawUrl.startsWith('http') ? rawUrl : undefined,
            gradcamUrl,
            gradcamPath:
              rawGradcamUrl && !rawGradcamUrl.startsWith('http') ? rawGradcamUrl : undefined,
            allScores: (row.all_scores as AnalysisHistoryItem['allScores']) ?? undefined,
            notes: (row.notes as string | null) ?? undefined,
            imageDecryptFailed:
              (rawUrl && !rawUrl.startsWith('http') && failedPaths.has(rawUrl)) ||
              (rawGradcamUrl && !rawGradcamUrl.startsWith('http') && failedPaths.has(rawGradcamUrl)) ||
              false,
            trustRecommendation: (row.trust_recommendation as string | null) ?? undefined,
            trustUncertaintyScore: (row.trust_uncertainty_score as number | null) ?? undefined,
            trustQualityFlags: (row.trust_quality_flags as string[] | null) ?? undefined,
          };
        });

        setHistoryCache({
          data: mappedRows,
          timestamp: Date.now(),
          loading: false,
          error: false,
        });

        return mappedRows;
      } catch (err) {
        console.error('Failed to fetch history cache:', err);
        setHistoryCache((prev) => ({ ...prev, loading: false, error: true }));
        return null;
      }
    },
    [historyCache.data, historyCache.timestamp, isCacheStale, userId]
  );

  // Cache invalidation methods
  const invalidateDashboard = useCallback(() => {
    setDashboardStats((prev) => ({ ...prev, timestamp: null }));
  }, []);

  const invalidateHistory = useCallback(() => {
    setHistoryCache((prev) => ({ ...prev, timestamp: null }));
  }, []);

  const invalidateAll = useCallback(() => {
    invalidateDashboard();
    invalidateHistory();
  }, [invalidateDashboard, invalidateHistory]);

  const value: DataCacheContextType = {
    dashboardStats,
    fetchDashboardStats,
    historyCache,
    fetchHistoryCache,
    invalidateAll,
    invalidateDashboard,
    invalidateHistory,
    userId,
    userName,
  };

  return <DataCacheContext.Provider value={value}>{children}</DataCacheContext.Provider>;
};

export const useDataCache = (): DataCacheContextType => {
  const context = useContext(DataCacheContext);
  if (!context) {
    throw new Error('useDataCache must be used within DataCacheProvider');
  }
  return context;
};
