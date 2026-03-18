
import React, { useState } from 'react';
import Input from './ui/Input';
import Button from './ui/Button';
import { supabase } from '../lib/supabase';
import { friendlyAuthError } from '../lib/authErrors';

interface ResetPasswordScreenProps {
  onPasswordReset: () => void;
}

const ResetPasswordScreen: React.FC<ResetPasswordScreenProps> = ({ onPasswordReset }) => {
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
      <div className="flex-1 flex flex-col items-center justify-center p-6 sm:p-12">
        <div className="w-full max-w-md bg-white rounded-2xl shadow-sm border border-slate-100 p-8 sm:p-10">
          <div className="text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-green-50 rounded-full mb-5">
              <svg className="w-8 h-8 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <h2 className="text-2xl font-bold tracking-tight text-slate-900 mb-2">
              Password Updated!
            </h2>
            <p className="text-slate-500 text-sm mb-8 leading-relaxed">
              Your password has been successfully reset. Please log in with your new password.
            </p>
            <Button onClick={onPasswordReset}>
              Go to Login
            </Button>
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
                d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" 
              />
            </svg>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 mb-2">
            Set New Password
          </h1>
          <p className="text-slate-500 text-sm px-4">
            Enter your new password below. It must be at least 12 characters.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-1">
          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-100 text-red-600 text-xs rounded-lg text-center font-medium">
              {error}
            </div>
          )}
          <Input 
            label="New Password" 
            type="password" 
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <Input 
            label="Confirm New Password" 
            type="password" 
            placeholder="••••••••"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
          />
          
          <div className="pt-4">
            <Button type="submit" disabled={loading}>
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Updating...
                </span>
              ) : 'Reset Password'}
            </Button>
          </div>
        </form>
      </div>

      <footer className="mt-12 text-center max-w-sm px-4">
        <p className="text-[11px] font-medium text-slate-400 uppercase tracking-widest leading-relaxed">
          Clinical Support Tool. Designed to assist medical professionals.
        </p>
      </footer>
    </div>
  );
};

export default ResetPasswordScreen;
