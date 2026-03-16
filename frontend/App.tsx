
import React, { useState, useEffect, useRef, lazy, Suspense } from 'react';
import { Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';
import { supabase } from './lib/supabase';
import type { ClassResult, AnalysisHistoryItem } from './lib/types';
import ErrorBoundary from './components/ErrorBoundary';
import AppLayout from './components/AppLayout';

// ── Code-split screen imports ─────────────────────────────────────────────────
const LoginScreen          = lazy(() => import('./components/LoginScreen'));
const SignupScreen         = lazy(() => import('./components/SignupScreen'));
const ForgotPasswordScreen = lazy(() => import('./components/ForgotPasswordScreen'));
const ResetPasswordScreen  = lazy(() => import('./components/ResetPasswordScreen'));
const DashboardScreen      = lazy(() => import('./components/DashboardScreen'));
const UploadScreen         = lazy(() => import('./components/UploadScreen'));
const ProcessingScreen     = lazy(() => import('./components/ProcessingScreen'));
const ResultsScreen        = lazy(() => import('./components/ResultsScreen'));
const HistoryScreen        = lazy(() => import('./components/HistoryScreen'));
const HistoryDetailScreen  = lazy(() => import('./components/HistoryDetailScreen'));
const ErrorScreen          = lazy(() => import('./components/ErrorScreen'));
const AboutScreen          = lazy(() => import('./components/AboutScreen'));
const HelpScreen           = lazy(() => import('./components/HelpScreen'));
const LogoutConfirmScreen  = lazy(() => import('./components/LogoutConfirmScreen'));
const ProfileScreen        = lazy(() => import('./components/ProfileScreen'));
const EmailVerificationScreen = lazy(() => import('./components/EmailVerificationScreen'));

// ── Route paths ───────────────────────────────────────────────────────────────
export const ROUTES = {
  login:             '/login',
  signup:            '/signup',
  forgotPassword:    '/forgot-password',
  resetPassword:     '/reset-password',
  emailVerification: '/email-verification',
  dashboard:         '/dashboard',
  upload:            '/upload',
  processing:        '/processing',
  results:           '/results',
  history:           '/history',
  historyDetail:     '/history/detail',
  error:             '/error',
  about:             '/about',
  help:              '/help',
  profile:           '/profile',
} as const;

const PUBLIC_ROUTES: string[] = [
  '/login', '/signup', '/forgot-password', '/reset-password', '/email-verification',
];

// ── Loading fallback ──────────────────────────────────────────────────────────
const PageLoader = () => (
  <div className="flex-1 flex items-center justify-center min-h-screen">
    <div className="text-center">
      <div className="inline-flex items-center justify-center w-12 h-12 bg-teal-50 rounded-xl mb-4">
        <svg className="animate-spin h-6 w-6 text-teal-600" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      </div>
      <p className="text-sm text-slate-400">Loading…</p>
    </div>
  </div>
);

// ── App ───────────────────────────────────────────────────────────────────────
const App: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const [selectedImage,       setSelectedImage]       = useState<string | null>(null);
  const [analysisResults,     setAnalysisResults]     = useState<ClassResult[] | null>(null);
  const [analysisError,       setAnalysisError]       = useState<string | null>(null);
  const [analysisRetryable,   setAnalysisRetryable]   = useState(false);
  const [selectedHistoryItem, setSelectedHistoryItem] = useState<AnalysisHistoryItem | null>(null);
  const [showLogoutConfirm,   setShowLogoutConfirm]   = useState(false);
  const [prevPath,            setPrevPath]            = useState<string | null>(null);
  const [authChecked,         setAuthChecked]         = useState(false);
  const [signupEmail,         setSignupEmail]         = useState<string>('');

  // ── Focus management ──────────────────────────────────────────────────────
  const prevPathRef = useRef<string>('');
  useEffect(() => {
    if (location.pathname === prevPathRef.current) return;
    prevPathRef.current = location.pathname;
    requestAnimationFrame(() => {
      const el = document.querySelector<HTMLElement>('h1, h2, [data-autofocus]');
      if (el) {
        el.setAttribute('tabindex', '-1');
        el.focus({ preventScroll: false });
        el.addEventListener('blur', () => el.removeAttribute('tabindex'), { once: true });
      }
    });
  }, [location.pathname]);

  // ── Auth init ─────────────────────────────────────────────────────────────
  useEffect(() => {
    const hashParams = new URLSearchParams(window.location.hash.substring(1));
    const isRecovery = hashParams.get('type') === 'recovery';

    supabase.auth.getSession().then(({ data: { session } }) => {
      if (isRecovery && session) {
        navigate(ROUTES.resetPassword, { replace: true });
      } else if (session && PUBLIC_ROUTES.includes(location.pathname)) {
        navigate(ROUTES.dashboard, { replace: true });
      } else if (!session && !PUBLIC_ROUTES.includes(location.pathname)) {
        navigate(ROUTES.login, { replace: true });
      }
      setAuthChecked(true);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((event) => {
      if (event === 'PASSWORD_RECOVERY') navigate(ROUTES.resetPassword);
      if (event === 'SIGNED_OUT')        navigate(ROUTES.login);
    });

    return () => subscription.unsubscribe();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleRequestLogout = () => {
    setPrevPath(location.pathname);
    setShowLogoutConfirm(true);
  };

  const handleConfirmLogout = async () => {
    setShowLogoutConfirm(false);
    setSelectedImage(null);
    setSelectedHistoryItem(null);
    await supabase.auth.signOut();
  };

  const handleCancelLogout = () => {
    setShowLogoutConfirm(false);
    if (prevPath) navigate(prevPath);
  };

  const resetAnalysis = () => {
    setSelectedImage(null);
    setAnalysisResults(null);
    setAnalysisError(null);
    setAnalysisRetryable(false);
    navigate(ROUTES.upload);
  };

  const handleViewHistoryDetail = (item: AnalysisHistoryItem) => {
    setSelectedHistoryItem(item);
    navigate(ROUTES.historyDetail);
  };

  if (!authChecked) return <PageLoader />;

  return (
    <ErrorBoundary>
      <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900 font-sans">
        <Suspense fallback={<PageLoader />}>
          <Routes>
            {/* ── Public ── */}
            <Route path={ROUTES.login} element={
              <LoginScreen
                onNavigateToSignup={() => navigate(ROUTES.signup)}
                onNavigateToForgotPassword={() => navigate(ROUTES.forgotPassword)}
                onLoginSuccess={() => navigate(ROUTES.dashboard)}
              />
            } />
            <Route path={ROUTES.signup} element={
              <SignupScreen
                onNavigateToLogin={() => navigate(ROUTES.login)}
                onSignupSuccess={(email) => {
                  setSignupEmail(email);
                  navigate(ROUTES.emailVerification);
                }}
              />
            } />
            <Route path={ROUTES.forgotPassword} element={
              <ForgotPasswordScreen onNavigateToLogin={() => navigate(ROUTES.login)} />
            } />
            <Route path={ROUTES.resetPassword} element={
              <ResetPasswordScreen onPasswordReset={() => navigate(ROUTES.login)} />
            } />
            <Route path={ROUTES.emailVerification} element={
              <EmailVerificationScreen
                email={signupEmail}
                onNavigateToLogin={() => navigate(ROUTES.login)}
                onResendEmail={async () => {
                  if (!signupEmail) throw new Error('No email address available to resend to.');
                  const { error } = await supabase.auth.resend({ type: 'signup', email: signupEmail });
                  if (error) throw error;
                }}
              />
            } />

            {/* ── Protected (all share the AppLayout sidebar) ── */}
            <Route path={ROUTES.dashboard} element={
              <AppLayout onLogout={handleRequestLogout}>
                <DashboardScreen
                  onNavigateToUpload={() => navigate(ROUTES.upload)}
                  onNavigateToHistory={() => navigate(ROUTES.history)}
                />
              </AppLayout>
            } />
            <Route path={ROUTES.upload} element={
              <AppLayout onLogout={handleRequestLogout}>
                <UploadScreen
                  selectedImage={selectedImage}
                  onImageSelect={setSelectedImage}
                  onBack={() => navigate(ROUTES.dashboard)}
                  onRunClassification={() => navigate(ROUTES.processing)}
                  onError={(msg) => { setAnalysisError(msg ?? null); navigate(ROUTES.error); }}
                />
              </AppLayout>
            } />
            <Route path={ROUTES.processing} element={
              <AppLayout onLogout={handleRequestLogout}>
                <ProcessingScreen
                  image={selectedImage}
                  onComplete={(results) => { setAnalysisResults(results); navigate(ROUTES.results); }}
                  onError={(msg, retryable) => { setAnalysisError(msg ?? null); setAnalysisRetryable(retryable ?? false); navigate(ROUTES.error); }}
                />
              </AppLayout>
            } />
            <Route path={ROUTES.results} element={
              <AppLayout onLogout={handleRequestLogout}>
                <ResultsScreen
                  image={selectedImage}
                  results={analysisResults}
                  onAnalyzeAnother={resetAnalysis}
                  onNavigateToHistory={() => navigate(ROUTES.history)}
                />
              </AppLayout>
            } />
            <Route path={ROUTES.history} element={
              <AppLayout onLogout={handleRequestLogout}>
                <HistoryScreen
                  onBack={() => navigate(ROUTES.dashboard)}
                  onViewDetails={handleViewHistoryDetail}
                />
              </AppLayout>
            } />
            <Route path={ROUTES.historyDetail} element={
              <AppLayout onLogout={handleRequestLogout}>
                <HistoryDetailScreen
                  item={selectedHistoryItem}
                  onBack={() => navigate(ROUTES.history)}
                />
              </AppLayout>
            } />
            <Route path={ROUTES.error} element={
              <AppLayout onLogout={handleRequestLogout}>
                <ErrorScreen
                  onBackToUpload={resetAnalysis}
                  onRetry={analysisRetryable && selectedImage ? () => { setAnalysisError(null); setAnalysisRetryable(false); navigate(ROUTES.processing); } : undefined}
                  message={analysisError ?? undefined}
                />
              </AppLayout>
            } />
            <Route path={ROUTES.about} element={
              <AppLayout onLogout={handleRequestLogout}>
                <AboutScreen onBack={() => navigate(ROUTES.dashboard)} />
              </AppLayout>
            } />
            <Route path={ROUTES.help} element={
              <AppLayout onLogout={handleRequestLogout}>
                <HelpScreen onBack={() => navigate(ROUTES.dashboard)} />
              </AppLayout>
            } />
            <Route path={ROUTES.profile} element={
              <AppLayout onLogout={handleRequestLogout}>
                <ProfileScreen onBack={() => navigate(ROUTES.dashboard)} />
              </AppLayout>
            } />

            {/* Fallbacks */}
            <Route path="/"  element={<Navigate to={ROUTES.login} replace />} />
            <Route path="*"  element={<Navigate to={ROUTES.login} replace />} />
          </Routes>
        </Suspense>

        {/* Logout confirmation overlay */}
        {showLogoutConfirm && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4">
            <Suspense fallback={null}>
              <LogoutConfirmScreen
                onConfirm={handleConfirmLogout}
                onCancel={handleCancelLogout}
              />
            </Suspense>
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
};

export default App;
