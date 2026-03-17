import React, { useState } from 'react';
import Input from './ui/Input';
import Button from './ui/Button';
import { supabase } from '../lib/supabase';

interface ForgotPasswordScreenProps {
  onNavigateToLogin: () => void;
}

const ForgotPasswordScreen: React.FC<ForgotPasswordScreenProps> = ({ onNavigateToLogin }) => {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!email.trim()) {
      setError('Please enter your email address.');
      return;
    }

    setLoading(true);
    try {
      const { error: resetError } = await supabase.auth.resetPasswordForEmail(
        email.trim(),
        {
          redirectTo: `${window.location.origin}/reset-password`,
        }
      );

      if (resetError) {
        setError(resetError.message);
      } else {
        setSent(true);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (sent) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 sm:p-12">
        <div className="w-full max-w-md bg-white rounded-2xl shadow-sm border border-slate-100 p-8 sm:p-10">
          <div className="text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-teal-50 rounded-full mb-5">
              <svg className="w-8 h-8 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
            </div>
            <h2 className="text-2xl font-bold tracking-tight text-slate-900 mb-2">
              Check Your Email
            </h2>
            <p className="text-slate-500 text-sm mb-2 leading-relaxed">
              We've sent a password reset link to:
            </p>
            <p className="text-teal-600 font-medium text-sm mb-6">{email}</p>
            <p className="text-slate-400 text-xs mb-8 leading-relaxed">
              If you don't see the email, check your spam folder. The link will expire in 24 hours.
            </p>
            <div className="flex flex-col gap-3">
              <Button onClick={onNavigateToLogin}>
                Back to Login
              </Button>
              <button
                onClick={() => { setSent(false); setEmail(''); }}
                className="text-sm font-medium text-slate-500 hover:text-teal-600 transition-colors"
              >
                Try a different email
              </button>
            </div>
          </div>
        </div>

        <footer className="mt-12 text-center max-w-sm px-4">
          <p className="text-[11px] font-medium text-slate-400 uppercase tracking-widest leading-relaxed">
            Clinical Support Tool. Designed to assist medical professionals.
          </p>
        </footer>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 sm:p-12">
      <div className="w-full max-w-md bg-white rounded-2xl shadow-sm border border-slate-100 p-8 sm:p-10">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-12 h-12 bg-teal-50 rounded-xl mb-4">
            <svg 
              className="w-6 h-6 text-teal-600" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24" 
              xmlns="http://www.w3.org/2000/svg"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" 
              />
            </svg>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 mb-2">
            Forgot Password
          </h1>
          <p className="text-slate-500 text-sm px-4">
            Enter your email and we'll send you a link to reset your password.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-100 text-red-600 text-xs rounded-lg text-center font-medium">
              {error}
            </div>
          )}
          <Input 
            label="Email Address" 
            type="email" 
            placeholder="Enter your registered email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          
          <div className="pt-2">
            <Button type="submit" disabled={loading}>
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Sending...
                </span>
              ) : 'Send Reset Link'}
            </Button>
          </div>
        </form>

        <div className="mt-8">
          <button 
            onClick={onNavigateToLogin}
            disabled={loading}
            className="w-full text-sm font-medium text-slate-500 hover:text-teal-600 transition-colors text-center flex items-center justify-center gap-2 disabled:opacity-50"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Login
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

export default ForgotPasswordScreen;
