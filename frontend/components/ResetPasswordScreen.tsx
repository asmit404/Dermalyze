import React, { useState } from 'react';
import { ShieldCheckIcon } from '@heroicons/react/24/outline';
import Input from './ui/Input';
import Button from './ui/Button';
import { supabase } from '../lib/supabase';
import { friendlyAuthError } from '../lib/authErrors';

interface ResetPasswordScreenProps {
  onPasswordReset: () => void;
  onNavigateToHome: () => void;
}

const ResetPasswordScreen: React.FC<ResetPasswordScreenProps> = ({
  onPasswordReset,
  onNavigateToHome,
}) => {
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (password.length < 12) {
      setError('Password must be at least 12 characters long.');
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    setLoading(true);
    try {
      const { error: updateError } = await supabase.auth.updateUser({
        password,
      });

      if (updateError) {
        setError(friendlyAuthError(updateError.message));
      } else {
        setSuccess(true);
        // Sign out so the user must log in with the new password
        await supabase.auth.signOut();
      }
    } catch {
      setError('An unexpected error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (success) {
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
                onClick={onPasswordReset}
                className="px-4 py-2 text-sm font-semibold text-white bg-teal-600 hover:bg-teal-700 rounded-lg shadow-sm transition-colors"
              >
                Sign In
              </button>
            </div>
          </div>
        </header>

        {/* Success Content */}
        <div className="flex-1 flex items-center justify-center px-6 py-12">
          <div className="w-full max-w-md">
            <div className="bg-white rounded-2xl border border-slate-300 shadow-sm p-10">
              <div className="text-center">
                <div className="inline-flex items-center justify-center w-16 h-16 bg-emerald-50 rounded-2xl mb-6">
                  <svg
                    className="w-10 h-10 text-emerald-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                </div>

                <h2 className="text-2xl font-bold text-slate-900 mb-3">Password Updated</h2>

                <p className="text-slate-600 mb-6 leading-relaxed">
                  Your password has been successfully reset. Please sign in with your new password.
                </p>

                <div className="bg-teal-50 border border-teal-200 rounded-xl p-4 mb-8">
                  <p className="text-sm text-teal-800 leading-relaxed">
                    <strong>Security tip:</strong> Store your new password in a secure password
                    manager.
                  </p>
                </div>

                <Button onClick={onPasswordReset} className="shadow-sm">
                  Continue to Sign In
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      {/* Header */}
      <header className="w-full bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
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
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-md">
          {/* Reset Password Card */}
          <div className="bg-white rounded-2xl border border-slate-300 shadow-sm p-10">
            <div className="mb-8">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-12 h-12 bg-teal-50 rounded-xl flex items-center justify-center">
                  <svg
                    className="w-6 h-6 text-teal-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                    />
                  </svg>
                </div>
              </div>
              <h2 className="text-2xl font-bold text-slate-900 mb-2">Set New Password</h2>
              <p className="text-sm text-slate-600">
                Enter your new password below. It must be at least 12 characters.
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-5">
              {error && (
                <div className="p-4 bg-red-50 border border-red-200 text-red-700 text-sm rounded-xl">
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

              <Input
                label="New Password"
                type="password"
                placeholder="Minimum 12 characters"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />

              <Input
                label="Confirm New Password"
                type="password"
                placeholder="Confirm password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />

              <Button type="submit" disabled={loading} className="shadow-sm">
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
                    Updating...
                  </span>
                ) : (
                  'Reset Password'
                )}
              </Button>
            </form>
          </div>

          {/* Footer Note */}
          <p className="mt-8 text-center text-xs text-slate-400 leading-relaxed px-4">
            By using this service, you confirm that you are a qualified medical professional using
            this system in accordance with professional guidelines.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ResetPasswordScreen;
