import React, { useState, useEffect } from 'react';
import { ShieldCheckIcon } from '@heroicons/react/24/outline';
import Input from './ui/Input';
import Button from './ui/Button';
import { supabase } from '../lib/supabase';
import { friendlyAuthError } from '../lib/authErrors';

interface LoginScreenProps {
  onNavigateToSignup: () => void;
  onNavigateToForgotPassword: () => void;
  onNavigateToHome: () => void;
  onLoginSuccess: () => void;
}

const MAX_ATTEMPTS = 5;
// 5-minute lockout — UX convenience only. Primary brute-force protection is
// Supabase server-side rate limiting (Dashboard → Auth → Rate Limits).
const LOCKOUT_MS = 5 * 60 * 1_000;
const LOCKOUT_KEY = 'dermalyze_login_lockout';

function readLockout(): { failCount: number; lockedUntil: number | null } {
  try {
    const raw = sessionStorage.getItem(LOCKOUT_KEY);
    if (!raw) return { failCount: 0, lockedUntil: null };
    const parsed = JSON.parse(raw) as { failCount: number; lockedUntil: number };
    if (Date.now() >= parsed.lockedUntil) {
      sessionStorage.removeItem(LOCKOUT_KEY);
      return { failCount: 0, lockedUntil: null };
    }
    return parsed;
  } catch {
    return { failCount: 0, lockedUntil: null };
  }
}

function persistLockout(failCount: number, lockedUntil: number): void {
  sessionStorage.setItem(LOCKOUT_KEY, JSON.stringify({ failCount, lockedUntil }));
}

const LoginScreen: React.FC<LoginScreenProps> = ({
  onNavigateToSignup,
  onNavigateToForgotPassword,
  onNavigateToHome,
  onLoginSuccess,
}) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [failCount, setFailCount] = useState(() => readLockout().failCount);
  const [lockedUntil, setLockedUntil] = useState<number | null>(() => readLockout().lockedUntil);
  const [lockCountdown, setLockCountdown] = useState(() => {
    const lu = readLockout().lockedUntil;
    return lu ? Math.ceil((lu - Date.now()) / 1000) : 0;
  });

  useEffect(() => {
    if (!lockedUntil) return;
    const interval = setInterval(() => {
      const remaining = Math.ceil((lockedUntil - Date.now()) / 1000);
      if (remaining <= 0) {
        setLockedUntil(null);
        setFailCount(0);
        setLockCountdown(0);
        setError('');
        sessionStorage.removeItem(LOCKOUT_KEY);
      } else {
        setLockCountdown(remaining);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [lockedUntil]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!email.trim() || !password.trim()) {
      setError('Please enter both email and password.');
      return;
    }

    if (lockedUntil && Date.now() < lockedUntil) return;

    setLoading(true);
    try {
      const { error: authError } = await supabase.auth.signInWithPassword({
        email: email.trim(),
        password,
      });

      if (authError) {
        const next = failCount + 1;
        setFailCount(next);
        if (next >= MAX_ATTEMPTS) {
          const until = Date.now() + LOCKOUT_MS;
          setLockedUntil(until);
          persistLockout(next, until);
          setLockCountdown(Math.ceil(LOCKOUT_MS / 1000));
          setError(`Too many failed attempts. Please wait ${LOCKOUT_MS / 60_000} minutes.`);
        } else if (authError.message.includes('Invalid login credentials')) {
          setError(
            `Invalid email or password. ${MAX_ATTEMPTS - next} attempt${MAX_ATTEMPTS - next === 1 ? '' : 's'} remaining.`
          );
        } else if (authError.message.includes('Email not confirmed')) {
          setError(
            'Please verify your email before logging in. Check your inbox for a confirmation link.'
          );
        } else {
          setError(friendlyAuthError(authError.message));
        }
      } else {
        onLoginSuccess();
      }
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : 'An unexpected error occurred. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      {/* Header */}
      <header className="w-full bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <button
            onClick={onNavigateToHome}
            className="flex items-center gap-3 hover:opacity-80 transition-opacity"
          >
            <div className="w-11 h-11 bg-teal-600 rounded-xl flex items-center justify-center shadow-sm">
              <ShieldCheckIcon className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-900 tracking-tight leading-tight">
                Dermalyze
              </h1>
              <p className="text-[9px] text-slate-500 uppercase tracking-wider font-semibold leading-tight">
                Clinical Decision Support
              </p>
            </div>
          </button>
          <div className="flex items-center gap-3">
            <button
              onClick={onNavigateToHome}
              className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-colors"
            >
              Home
            </button>
            <button
              onClick={onNavigateToSignup}
              className="px-4 py-2 text-sm font-semibold text-white bg-teal-600 hover:bg-teal-700 rounded-lg shadow-sm transition-colors"
            >
              Create Account
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-md">
          {/* Login Card */}
          <div className="bg-white rounded-2xl border border-slate-300 shadow-sm p-10">
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-slate-900 mb-2">Sign In</h2>
              <p className="text-sm text-slate-600">
                Access your dermatology classification workspace
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-5">
              {error && (
                <div
                  role="alert"
                  aria-live="assertive"
                  className="p-4 bg-red-50 border border-red-200 text-red-700 text-sm rounded-xl"
                >
                  <div className="flex items-start gap-3">
                    <svg
                      className="w-5 h-5 shrink-0 mt-0.5"
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
                    <span>{error}</span>
                  </div>
                </div>
              )}
              {lockedUntil && (
                <div
                  role="status"
                  aria-live="polite"
                  className="p-3 bg-amber-50 border border-amber-200 text-amber-800 text-sm rounded-xl text-center font-medium"
                >
                  Account temporarily locked — retry in {lockCountdown}s
                </div>
              )}

              <Input
                label="Email Address"
                type="email"
                placeholder="Email address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />

              <Input
                label="Password"
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />

              <div className="flex items-center justify-end">
                <button
                  type="button"
                  onClick={onNavigateToForgotPassword}
                  className="text-sm text-teal-600 hover:text-teal-700 font-medium transition-colors"
                >
                  Forgot password?
                </button>
              </div>

              <Button
                type="submit"
                disabled={loading || !!(lockedUntil && Date.now() < lockedUntil)}
                className="shadow-sm"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
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
                    Signing in...
                  </span>
                ) : (
                  'Sign In'
                )}
              </Button>
            </form>

            <div className="mt-8 pt-6 border-t border-slate-200">
              <p className="text-center text-sm text-slate-600">
                Don't have an account?{' '}
                <button
                  onClick={onNavigateToSignup}
                  className="font-semibold text-teal-600 hover:text-teal-700 transition-colors"
                >
                  Create Account
                </button>
              </p>
            </div>
          </div>

          {/* Footer Note */}
          <p className="mt-8 text-center text-xs text-slate-400 leading-relaxed px-4">
            By signing in, you confirm that you are a qualified medical professional using this
            system in accordance with professional guidelines.
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoginScreen;
