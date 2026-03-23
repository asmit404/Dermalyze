
import React, { useState } from 'react';
import Input from './ui/Input';
import Button from './ui/Button';
import { supabase } from '../lib/supabase';
import { friendlyAuthError } from '../lib/authErrors';

interface SignupScreenProps {
  onNavigateToLogin: () => void;
  onSignupSuccess?: (email: string) => void;
}

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

type FieldKey = 'email' | 'password' | 'confirmPassword';

const SignupScreen: React.FC<SignupScreenProps> = ({ onNavigateToLogin, onSignupSuccess }) => {
  const [name,            setName]            = useState('');
  const [email,           setEmail]           = useState('');
  const [password,        setPassword]        = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [serverError,     setServerError]     = useState('');
  const [loading,         setLoading]         = useState(false);
  const [success,         setSuccess]         = useState(false);

  const [fieldErrors, setFieldErrors] = useState<Record<FieldKey, string>>({
    email: '', password: '', confirmPassword: '',
  });
  const [touched, setTouched] = useState<Record<FieldKey, boolean>>({
    email: false, password: false, confirmPassword: false,
  });

  // Returns the error string for a field given its current value
  const validate = (field: FieldKey, value: string, currentPassword = password): string => {
    switch (field) {
      case 'email':
        if (!value.trim()) return 'Please enter your email address.';
        if (!EMAIL_RE.test(value.trim())) return 'Please enter a valid email address.';
        return '';
      case 'password':
        if (value.length > 0 && value.length < 12) return 'Password must be at least 12 characters.';
        return '';
      case 'confirmPassword':
        if (value && value !== currentPassword) return 'Passwords do not match.';
        return '';
    }
  };

  const handleBlur = (field: FieldKey) => {
    const value = field === 'email' ? email : field === 'password' ? password : confirmPassword;
    setTouched(prev => ({ ...prev, [field]: true }));
    setFieldErrors(prev => ({ ...prev, [field]: validate(field, value) }));
  };

  const handleEmailChange = (v: string) => {
    setEmail(v);
    if (touched.email) setFieldErrors(prev => ({ ...prev, email: validate('email', v) }));
  };

  const handlePasswordChange = (v: string) => {
    setPassword(v);
    if (touched.password) setFieldErrors(prev => ({ ...prev, password: validate('password', v) }));
    // Re-check confirm if it's already been touched
    if (touched.confirmPassword)
      setFieldErrors(prev => ({ ...prev, confirmPassword: validate('confirmPassword', confirmPassword, v) }));
  };

  const handleConfirmChange = (v: string) => {
    setConfirmPassword(v);
    if (touched.confirmPassword)
      setFieldErrors(prev => ({ ...prev, confirmPassword: validate('confirmPassword', v) }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setServerError('');

    // Touch all fields so errors become visible
    const allTouched: Record<FieldKey, boolean> = { email: true, password: true, confirmPassword: true };
    setTouched(allTouched);

    const emailErr   = validate('email',           email);
    const pwErr      = validate('password',        password);
    const confirmErr = validate('confirmPassword', confirmPassword);

    // Also enforce that both password fields are non-empty on submit
    const pwRequired      = !password          ? 'Please enter a password.'         : pwErr;
    const confirmRequired = !confirmPassword   ? 'Please confirm your password.'    : confirmErr;

    setFieldErrors({ email: emailErr, password: pwRequired, confirmPassword: confirmRequired });

    if (emailErr || pwRequired || confirmRequired) return;

    setLoading(true);
    try {
      const { data, error: authError } = await supabase.auth.signUp({
        email: email.trim(),
        password,
        options: {
          data: {
            full_name: name.trim() || undefined,
          },
        },
      });

      if (authError) {
        if (authError.message.includes('valid email')) {
          setServerError('Please enter a valid email address.');
        } else {
          setServerError(friendlyAuthError(authError.message));
        }
      } else if (data.user?.identities?.length === 0) {
        // Supabase silently "succeeds" for existing emails to prevent enumeration.
        // An empty identities array is the reliable signal that the email is taken.
        setServerError('This email is already registered. Please log in instead.');
      } else {
        if (onSignupSuccess) {
          onSignupSuccess(email.trim());
        } else {
          setSuccess(true);
        }
      }
    } catch (err: unknown) {
      setServerError('An unexpected error occurred. Please try again.');
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
              Account Created!
            </h2>
            <p className="text-slate-500 text-sm mb-2 leading-relaxed">
              Your account has been created successfully.
            </p>
            <p className="text-slate-500 text-sm mb-8 leading-relaxed">
              Please check your email inbox for a verification link to activate your account before logging in.
            </p>
            <Button onClick={onNavigateToLogin}>
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
                d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" 
              />
            </svg>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 mb-1">
            Create Account
          </h1>
          <p className="text-slate-500 text-sm">
            Access our advanced skin lesion classification system
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-1">
          {serverError && (
            <div className="mb-4 p-3 bg-red-50 border border-red-100 text-red-600 text-xs rounded-lg text-center font-medium">
              {serverError}
            </div>
          )}
          <Input 
            label="Name (Optional)" 
            type="text" 
            placeholder="Enter your full name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <Input 
            label="Email" 
            type="email" 
            placeholder="Enter your email id"
            value={email}
            onChange={(e) => handleEmailChange(e.target.value)}
            onBlur={() => handleBlur('email')}
            error={fieldErrors.email || undefined}
          />
          <Input 
            label="Password" 
            type="password" 
            placeholder="••••••••"
            value={password}
            onChange={(e) => handlePasswordChange(e.target.value)}
            onBlur={() => handleBlur('password')}
            error={fieldErrors.password || undefined}
          />
          <Input 
            label="Confirm Password" 
            type="password" 
            placeholder="••••••••"
            value={confirmPassword}
            onChange={(e) => handleConfirmChange(e.target.value)}
            onBlur={() => handleBlur('confirmPassword')}
            error={fieldErrors.confirmPassword || undefined}
          />
          
          <div className="pt-4">
            <Button type="submit" disabled={loading}>
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Creating account...
                </span>
              ) : 'Create Account'}
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

export default SignupScreen;
