import React, { useState, useEffect } from 'react';
import Input from './ui/Input';
import Button from './ui/Button';
import { supabase } from '../lib/supabase';
import { UserIcon } from '@heroicons/react/24/outline';

interface ProfileScreenProps {
  onBack: () => void;
}

const ProfileScreen: React.FC<ProfileScreenProps> = ({ onBack }) => {
  // ── Profile state ───────────────────────────────────────────────────────────
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [profileLoading, setProfileLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [profileMsg, setProfileMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(
    null
  );

  // ── Change password state ───────────────────────────────────────────────────
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmNewPassword, setConfirmNewPassword] = useState('');
  const [passwordLoading, setPasswordLoading] = useState(false);
  const [passwordMsg, setPasswordMsg] = useState<{
    type: 'success' | 'error';
    text: string;
  } | null>(null);

  // ── Delete account state ────────────────────────────────────────────────────
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteConfirmText, setDeleteConfirmText] = useState('');
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [deleteMsg, setDeleteMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(
    null
  );

  // ── Load user data on mount ─────────────────────────────────────────────────
  useEffect(() => {
    const loadProfile = async () => {
      try {
        const {
          data: { user },
        } = await supabase.auth.getUser();
        if (user) {
          setEmail(user.email ?? '');
          setFullName(user.user_metadata?.full_name ?? '');
        }
      } catch {
        setProfileMsg({ type: 'error', text: 'Failed to load profile data.' });
      } finally {
        setProfileLoading(false);
      }
    };
    loadProfile();
  }, []);

  // ── Save profile changes ───────────────────────────────────────────────────
  const handleSaveProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    setProfileMsg(null);
    setSaving(true);
    try {
      const { error } = await supabase.auth.updateUser({
        data: { full_name: fullName.trim() },
      });
      if (error) throw error;
      setProfileMsg({ type: 'success', text: 'Profile updated successfully.' });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to update profile.';
      setProfileMsg({ type: 'error', text: message });
    } finally {
      setSaving(false);
    }
  };

  // ── Change password ─────────────────────────────────────────────────────────
  const handleChangePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setPasswordMsg(null);

    if (!currentPassword) {
      setPasswordMsg({ type: 'error', text: 'Please enter your current password.' });
      return;
    }
    if (newPassword.length < 12) {
      setPasswordMsg({ type: 'error', text: 'New password must be at least 12 characters.' });
      return;
    }
    if (newPassword !== confirmNewPassword) {
      setPasswordMsg({ type: 'error', text: 'Passwords do not match.' });
      return;
    }

    setPasswordLoading(true);
    try {
      // Verify current password before allowing the change
      const { error: verifyError } = await supabase.auth.signInWithPassword({
        email,
        password: currentPassword,
      });
      if (verifyError) {
        setPasswordMsg({ type: 'error', text: 'Current password is incorrect.' });
        setPasswordLoading(false);
        return;
      }

      const { error } = await supabase.auth.updateUser({ password: newPassword });
      if (error) throw error;
      setCurrentPassword('');
      setNewPassword('');
      setConfirmNewPassword('');
      setPasswordMsg({ type: 'success', text: 'Password changed successfully.' });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to change password.';
      setPasswordMsg({ type: 'error', text: message });
    } finally {
      setPasswordLoading(false);
    }
  };

  // ── Delete account ──────────────────────────────────────────────────────────
  const handleDeleteAccount = async () => {
    if (deleteConfirmText !== 'DELETE') return;
    setDeleteMsg(null);
    setDeleteLoading(true);
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session) throw new Error('Not authenticated');

      const userId = session.user.id;

      // 1. Delete user's analysis images from storage
      let hasMoreFiles = true;
      while (hasMoreFiles) {
        const { data: files, error: listError } = await supabase.storage
          .from('analysis-images')
          .list(userId, { limit: 100 });
          
        if (listError) throw new Error('Failed to list analysis images.');
        
        if (!files || files.length === 0) {
          hasMoreFiles = false;
          break;
        }
        
        const paths = files.map((f) => `${userId}/${f.name}`);
        const { error: removeError, data: removedData } = await supabase.storage
          .from('analysis-images')
          .remove(paths);
          
        if (removeError) throw new Error('Failed to delete analysis images.');
        
        if (!removedData || removedData.length === 0) {
          throw new Error('Failed to fully delete analysis images.');
        }

        if (files.length < 100) {
          hasMoreFiles = false;
        }
      }

      // 2. Call the Edge Function to fully delete the auth user.
      //    ON DELETE CASCADE on the analyses table auto-removes all records.
      const { error: fnError } = await supabase.functions.invoke('delete-user');

      if (fnError) {
        // FunctionsHttpError: real message lives in the response body, not fnError.message
        let errorMessage = 'Account deletion failed.';
        try {
          // context is the raw Response object when the function returns non-2xx
          const body = await (fnError as unknown as { context: Response }).context.json();
          if (body?.error) errorMessage = body.error;
          else if (
            fnError.message &&
            fnError.message !== 'Edge Function returned a non-2xx status code'
          ) {
            errorMessage = fnError.message;
          }
        } catch {
          // fall back to generic message
        }
        throw new Error(errorMessage);
      }

      // 3. Sign out locally (server-side user is already gone)
      await supabase.auth.signOut();

      // Navigation to /login is handled by the auth state change listener in App.
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to delete account.';
      setDeleteMsg({ type: 'error', text: message });
      setDeleteLoading(false);
    }
  };

  // ── Helpers ─────────────────────────────────────────────────────────────────
  const initials = fullName
    ? fullName
        .split(' ')
        .map((n) => n[0])
        .join('')
        .toUpperCase()
        .slice(0, 2)
    : email
      ? email[0].toUpperCase()
      : '?';

  const FeedbackBanner: React.FC<{ msg: { type: 'success' | 'error'; text: string } | null }> = ({
    msg,
  }) => {
    if (!msg) return null;
    const isSuccess = msg.type === 'success';
    return (
      <div
        role="alert"
        className={`mb-4 p-3 text-xs rounded-lg text-center font-medium border ${
          isSuccess
            ? 'bg-emerald-50 border-emerald-100 text-emerald-700'
            : 'bg-red-50 border-red-100 text-red-600'
        }`}
      >
        {msg.text}
      </div>
    );
  };

  // ── Skeleton ────────────────────────────────────────────────────────────────
  if (profileLoading) {
    return (
      <div className="flex-1 flex flex-col bg-slate-50 text-slate-900 min-h-screen">
        <main className="max-w-2xl mx-auto w-full px-4 sm:px-6 py-8 space-y-6">
          <div className="h-12 bg-white border border-slate-200 rounded-xl animate-pulse" />
          <div className="h-64 bg-white border border-slate-200 rounded-xl animate-pulse" />
          <div className="h-48 bg-white border border-slate-200 rounded-xl animate-pulse" />
        </main>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col bg-slate-50 text-slate-900 min-h-screen">
      {/* ── Header ── */}
      <section className="bg-gradient-to-br from-teal-600 to-teal-700 px-6 py-8 sm:py-10 shrink-0">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center justify-between">
            <div>
              <span className="inline-flex items-center gap-2 px-3 py-1.5 bg-white/20 backdrop-blur-sm text-white text-[11px] font-bold uppercase tracking-wider rounded-lg border border-white/30 mb-3">
                <UserIcon className="w-4 h-4" />
                Account Settings
              </span>
              <h1 className="text-3xl font-bold text-white tracking-tight mb-2">
                My Profile
              </h1>
              <p className="text-teal-50 text-sm leading-relaxed">
                Manage your identity, security credentials, and data preferences.
              </p>
            </div>
            <button
              onClick={onBack}
              className="inline-flex items-center gap-2 px-3 sm:px-5 py-2 sm:py-2.5 text-sm font-semibold text-white border-2 border-white/30 hover:bg-white/10 rounded-xl transition-colors shrink-0"
            >
              <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              <span className="hidden sm:inline">Back to Dashboard</span>
              <span className="inline sm:hidden">Back</span>
            </button>
          </div>
        </div>
      </section>

      {/* ── Main ── */}
      <main className="flex-1 flex flex-col min-h-0">
        <section className="bg-slate-100 px-6 py-8 flex-1 flex flex-col justify-center">
          <div className="max-w-5xl w-full mx-auto grid lg:grid-cols-2 gap-6 items-stretch">
            
            {/* ── Left Column: Password Management ── */}
            <div className="flex flex-col">
              <article className="bg-white rounded-3xl border-2 border-slate-300 p-6 sm:p-8 shadow-sm flex-1 flex flex-col">
                <div className="mb-8">
                  <h3 className="text-xl font-bold text-slate-900">Password Management</h3>
                  <p className="text-sm text-slate-500 mt-2">
                    Update your password to keep your account secure. Must be at least 12 characters.
                  </p>
                </div>

                <form onSubmit={handleChangePassword} className="space-y-5 flex-1 flex flex-col">
                  <FeedbackBanner msg={passwordMsg} />
                  <Input
                    label="Current Password"
                    type="password"
                    placeholder="••••••••"
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    required
                  />
                  <Input
                    label="New Password"
                    type="password"
                    placeholder="••••••••"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    required
                  />
                  <Input
                    label="Confirm Password"
                    type="password"
                    placeholder="••••••••"
                    value={confirmNewPassword}
                    onChange={(e) => setConfirmNewPassword(e.target.value)}
                    required
                  />
                  
                  {/* Pushes the button to the bottom if there's extra space */}
                  <div className="mt-auto pt-6 flex justify-end">
                    <div className="w-full sm:w-auto">
                      <Button type="submit" disabled={passwordLoading}>
                        {passwordLoading ? (
                          <span className="flex items-center justify-center gap-2">
                            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                            </svg>
                            Updating…
                          </span>
                        ) : (
                          'Update Password'
                        )}
                      </Button>
                    </div>
                  </div>
                </form>
              </article>
            </div>

            {/* ── Right Column: Identity & Danger Zone ── */}
            <div className="flex flex-col gap-6">
              
              {/* ── Personal Information ── */}
              <article className="bg-white rounded-3xl border-2 border-slate-300 p-6 sm:p-8 shadow-sm">
                <div className="flex flex-col sm:flex-row items-center sm:items-start gap-6 mb-6">
                  {/* Avatar */}
                  <div className="w-20 h-20 shrink-0 rounded-2xl bg-teal-100 border-4 border-white shadow-sm flex items-center justify-center text-teal-700 font-bold text-2xl select-none">
                    {initials}
                  </div>
                  {/* Info */}
                  <div className="text-center sm:text-left min-w-0 flex-1">
                    <h2 className="text-xl font-bold text-slate-900 break-words">{fullName || 'Unnamed User'}</h2>
                    <p className="text-sm text-slate-500 break-all mb-2">{email}</p>
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-slate-100 text-slate-400 uppercase tracking-widest">
                      Primary Email
                    </span>
                  </div>
                </div>

                <div className="border-t border-slate-200 pt-6">
                  <form onSubmit={handleSaveProfile} className="space-y-4">
                    <FeedbackBanner msg={profileMsg} />
                    <div className="flex flex-col sm:flex-row items-end gap-4">
                      <div className="flex-1 w-full [&>*]:mb-0">
                        <Input
                          label="Full Name"
                          type="text"
                          placeholder="Enter your full name"
                          value={fullName}
                          onChange={(e) => setFullName(e.target.value)}
                        />
                      </div>
                      <div className="w-full sm:w-auto">
                        <Button type="submit" disabled={saving}>
                          {saving ? (
                            <span className="flex items-center justify-center gap-2">
                              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                              </svg>
                              Saving…
                            </span>
                          ) : (
                            'Save'
                          )}
                        </Button>
                      </div>
                    </div>
                  </form>
                </div>
              </article>

              {/* ── Account Deletion ── */}
              <article className="bg-rose-50 rounded-3xl border-2 border-rose-300 p-6 sm:p-8 shadow-sm flex-1 flex flex-col">
                <div className="mb-5">
                  <h3 className="text-xl font-bold text-red-800">Account Deletion</h3>
                  <p className="text-sm text-red-700/80 mt-2 leading-relaxed">
                    Permanently delete your account. All of your analysis records and uploaded images will be permanently erased. This cannot be undone.
                  </p>
                </div>

                <div className="mt-auto pt-2">
                  {!showDeleteConfirm ? (
                    <button
                      onClick={() => setShowDeleteConfirm(true)}
                      className="px-6 py-2.5 rounded-xl text-sm font-semibold border-2 border-red-200 bg-white text-red-700 hover:bg-red-50 transition-colors shadow-sm"
                    >
                      Delete Account
                    </button>
                  ) : (
                    <div className="bg-white p-5 rounded-2xl border border-rose-200 shadow-sm">
                      <FeedbackBanner msg={deleteMsg} />
                      <p className="text-sm text-red-700 mb-3 font-semibold">
                        Permanent action. Type <span className="font-mono font-bold px-1.5 py-0.5 bg-rose-100 rounded text-red-800">DELETE</span> to confirm.
                      </p>
                      <div className="flex flex-col sm:flex-row gap-3">
                        <input
                          type="text"
                          value={deleteConfirmText}
                          onChange={(e) => setDeleteConfirmText(e.target.value)}
                          placeholder="Type DELETE"
                          className="flex-1 px-4 py-2.5 rounded-lg border border-rose-200 bg-slate-50 focus:outline-none focus:ring-2 focus:ring-red-500/20 focus:border-red-500 transition-all text-slate-800 placeholder:text-slate-400 text-sm font-mono"
                          autoComplete="off"
                        />
                        <button
                          onClick={handleDeleteAccount}
                          disabled={deleteConfirmText !== 'DELETE' || deleteLoading}
                          className="sm:w-32 py-2.5 px-4 rounded-lg font-semibold text-sm bg-red-600 text-white hover:bg-red-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm shrink-0"
                        >
                          {deleteLoading ? 'Deleting…' : 'Confirm'}
                        </button>
                        <button
                          onClick={() => {
                            setShowDeleteConfirm(false);
                            setDeleteConfirmText('');
                            setDeleteMsg(null);
                          }}
                          disabled={deleteLoading}
                          className="sm:w-28 py-2.5 px-4 rounded-lg font-semibold text-sm border-2 border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-colors disabled:opacity-50 shadow-sm shrink-0"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </article>

            </div>
          </div>
        </section>
      </main>

      {/* ── Footer ── */}
      <footer className="bg-white border-t-2 border-slate-300 py-6 px-6 shrink-0">
        <div className="max-w-6xl mx-auto text-center">
          <p className="text-[10px] sm:text-xs text-slate-400 uppercase tracking-widest font-bold">
            Clinical Support & Diagnostic Aid Suite
          </p>
        </div>
      </footer>
    </div>
  );
};

export default ProfileScreen;

