
import React, { useState, useEffect } from 'react';
import Input from './ui/Input';
import Button from './ui/Button';
import { supabase } from '../lib/supabase';
import { friendlyAuthError } from '../lib/authErrors';

interface LoginScreenProps {
  onNavigateToSignup: () => void;
  onNavigateToForgotPassword: () => void;
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
          setError(`Invalid email or password. ${MAX_ATTEMPTS - next} attempt${MAX_ATTEMPTS - next === 1 ? '' : 's'} remaining.`);
        } else if (authError.message.includes('Email not confirmed')) {
          setError('Please verify your email before logging in. Check your inbox for a confirmation link.');
        } else {
          setError(friendlyAuthError(authError.message));
        }
      } else {
        onLoginSuccess();
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 sm:p-12">
      <div className="w-full max-w-md bg-white rounded-2xl shadow-sm border border-slate-100 p-8 sm:p-10">
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-teal-50 rounded-2xl mb-4">
            <svg 
              className="w-8 h-8 text-teal-600" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24" 
              xmlns="http://www.w3.org/2000/svg"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={1.5} 
                d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" 
              />
            </svg>
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-900 mb-2">
            Dermalyze
          </h1>
          <p className="text-slate-500 text-sm">
            AI-Assisted Skin Lesion Classification System
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-2">
          {error && (
            <div role="alert" aria-live="assertive" className="mb-4 p-3 bg-red-50 border border-red-100 text-red-600 text-xs rounded-lg text-center font-medium">
              {error}
            </div>
          )}
          {lockedUntil && (
            <div role="status" aria-live="polite" className="mb-2 p-2 bg-amber-50 border border-amber-100 text-amber-700 text-xs rounded-lg text-center">
              Locked — try again in {lockCountdown}s
            </div>
          )}
          <Input 
            label="Email" 
            type="email" 
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <Input 
            label="Password" 
            type="password" 
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          
          <div className="pt-4">
            <Button type="submit" disabled={loading || !!(lockedUntil && Date.now() < lockedUntil)}>
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Logging in...
                </span>
              ) : 'Login'}
            </Button>
          </div>
        </form>

        <div className="mt-8 flex flex-col gap-3">
          <button 
            onClick={onNavigateToSignup}
            className="text-sm font-medium text-slate-600 hover:text-teal-600 transition-colors text-center"
          >
            Create an account
          </button>
          <button 
            onClick={onNavigateToForgotPassword}
            className="text-sm text-slate-400 hover:text-slate-600 transition-colors text-center"
          >
            Forgot password?
          </button>
        </div>
      </div>

      <footer className="mt-12 text-center max-w-sm px-4">
        <p className="text-[11px] font-medium text-slate-400 uppercase tracking-widest leading-relaxed">
          Clinical Support Tool. Designed to assist medical professionals.
        </p>
      </footer>
    </div>
  );
};

export default LoginScreen;
