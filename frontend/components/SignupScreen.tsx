import React, { useState } from 'react';
import { ShieldCheckIcon } from '@heroicons/react/24/outline';
import Input from './ui/Input';
import Button from './ui/Button';
import { supabase } from '../lib/supabase';
import { friendlyAuthError } from '../lib/authErrors';

interface SignupScreenProps {
  onNavigateToLogin: () => void;
  onNavigateToHome: () => void;
  onSignupSuccess?: (email: string) => void;
}

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

type FieldKey = 'email' | 'password' | 'confirmPassword';

const SignupScreen: React.FC<SignupScreenProps> = ({
  onNavigateToLogin,
  onNavigateToHome,
  onSignupSuccess,
}) => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [serverError, setServerError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const [fieldErrors, setFieldErrors] = useState<Record<FieldKey, string>>({
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [touched, setTouched] = useState<Record<FieldKey, boolean>>({
    email: false,
    password: false,
    confirmPassword: false,
  });

  // Returns the error string for a field given its current value
  const validate = (field: FieldKey, value: string, currentPassword = password): string => {
    switch (field) {
      case 'email':
        if (!value.trim()) return 'Please enter your email address.';
        if (!EMAIL_RE.test(value.trim())) return 'Please enter a valid email address.';
        return '';
      case 'password':
        if (value.length > 0 && value.length < 12)
          return 'Password must be at least 12 characters.';
        return '';
      case 'confirmPassword':
        if (value && value !== currentPassword) return 'Passwords do not match.';
        return '';
    }
  };

  const handleBlur = (field: FieldKey) => {
    const value = field === 'email' ? email : field === 'password' ? password : confirmPassword;
    setTouched((prev) => ({ ...prev, [field]: true }));
    setFieldErrors((prev) => ({ ...prev, [field]: validate(field, value) }));
  };

  const handleEmailChange = (v: string) => {
    setEmail(v);
    if (touched.email) setFieldErrors((prev) => ({ ...prev, email: validate('email', v) }));
  };

  const handlePasswordChange = (v: string) => {
    setPassword(v);
    if (touched.password)
      setFieldErrors((prev) => ({ ...prev, password: validate('password', v) }));
    // Re-check confirm if it's already been touched
    if (touched.confirmPassword)
      setFieldErrors((prev) => ({
        ...prev,
        confirmPassword: validate('confirmPassword', confirmPassword, v),
      }));
  };

  const handleConfirmChange = (v: string) => {
    setConfirmPassword(v);
    if (touched.confirmPassword)
      setFieldErrors((prev) => ({ ...prev, confirmPassword: validate('confirmPassword', v) }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setServerError('');

    // Touch all fields so errors become visible
    const allTouched: Record<FieldKey, boolean> = {
      email: true,
      password: true,
      confirmPassword: true,
    };
    setTouched(allTouched);

    const emailErr = validate('email', email);
    const pwErr = validate('password', password);
    const confirmErr = validate('confirmPassword', confirmPassword);

    // Also enforce that both password fields are non-empty on submit
    const pwRequired = !password ? 'Please enter a password.' : pwErr;
    const confirmRequired = !confirmPassword ? 'Please confirm your password.' : confirmErr;

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
                onClick={onNavigateToLogin}
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
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                </div>

                <h2 className="text-2xl font-bold text-slate-900 mb-3">
                  Account Created Successfully
                </h2>

                <p className="text-slate-600 mb-6 leading-relaxed">
                  We've sent a verification email to{' '}
                  <strong className="text-slate-900">{email}</strong>. Please check your inbox and
                  click the confirmation link to activate your account.
                </p>

                <div className="bg-teal-50 border border-teal-200 rounded-xl p-4 mb-8">
                  <p className="text-sm text-teal-800 leading-relaxed">
                    <strong>Next step:</strong> Verify your email address before signing in. The
                    link expires in 24 hours.
                  </p>
                </div>

                <Button onClick={onNavigateToLogin} className="shadow-sm">
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
        <div className="w-full max-w-2xl">
          {/* Signup Card */}
          <div className="bg-white rounded-2xl border border-slate-300 shadow-sm p-10">
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-slate-900 mb-2">Create Account</h2>
              <p className="text-sm text-slate-600">
                Set up your dermatology classification workspace
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-5">
              {serverError && (
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
                    <span>{serverError}</span>
                  </div>
                </div>
              )}

              {/* Row 1: Full Name + Email */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <Input
                  label="Full Name (Optional)"
                  type="text"
                  placeholder="Full name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />

                <Input
                  label="Email Address"
                  type="email"
                  placeholder="Email address"
                  value={email}
                  onChange={(e) => handleEmailChange(e.target.value)}
                  onBlur={() => handleBlur('email')}
                  error={fieldErrors.email || undefined}
                />
              </div>

              {/* Row 2: Password + Confirm Password */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <Input
                  label="Password"
                  type="password"
                  placeholder="Minimum 12 characters"
                  value={password}
                  onChange={(e) => handlePasswordChange(e.target.value)}
                  onBlur={() => handleBlur('password')}
                  error={fieldErrors.password || undefined}
                />

                <Input
                  label="Confirm Password"
                  type="password"
                  placeholder="Confirm password"
                  value={confirmPassword}
                  onChange={(e) => handleConfirmChange(e.target.value)}
                  onBlur={() => handleBlur('confirmPassword')}
                  error={fieldErrors.confirmPassword || undefined}
                />
              </div>

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
                    Creating account...
                  </span>
                ) : (
                  'Create Account'
                )}
              </Button>
            </form>

            <div className="mt-8 pt-6 border-t border-slate-200">
              <p className="text-center text-sm text-slate-600">
                Already have an account?{' '}
                <button
                  onClick={onNavigateToLogin}
                  disabled={loading}
                  className="font-semibold text-teal-600 hover:text-teal-700 transition-colors disabled:opacity-50"
                >
                  Sign In
                </button>
              </p>
            </div>
          </div>

          {/* Footer Note */}
          <p className="mt-8 text-center text-xs text-slate-400 leading-relaxed px-4">
            By creating an account, you confirm that you are a qualified medical professional and
            agree to use this system in accordance with professional guidelines.
          </p>
        </div>
      </div>
    </div>
  );
};

export default SignupScreen;
