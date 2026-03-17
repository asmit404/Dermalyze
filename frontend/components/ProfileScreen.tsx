
import React, { useState, useEffect } from 'react';
import Input from './ui/Input';
import Button from './ui/Button';
import { supabase } from '../lib/supabase';

interface ProfileScreenProps {
  onBack: () => void;
}

const ProfileScreen: React.FC<ProfileScreenProps> = ({ onBack }) => {
  // ── Profile state ───────────────────────────────────────────────────────────
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [profileLoading, setProfileLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [profileMsg, setProfileMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // ── Change password state ───────────────────────────────────────────────────
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmNewPassword, setConfirmNewPassword] = useState('');
  const [passwordLoading, setPasswordLoading] = useState(false);
  const [passwordMsg, setPasswordMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // ── Delete account state ────────────────────────────────────────────────────
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteConfirmText, setDeleteConfirmText] = useState('');
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [deleteMsg, setDeleteMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // ── Load user data on mount ─────────────────────────────────────────────────
  useEffect(() => {
    const loadProfile = async () => {
      try {
        const { data: { user } } = await supabase.auth.getUser();
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
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) throw new Error('Not authenticated');

      const userId = session.user.id;

      // 1. Delete user's analysis images from storage
      const { data: files } = await supabase.storage
        .from('analysis-images')
        .list(userId);
      if (files && files.length > 0) {
        const paths = files.map((f) => `${userId}/${f.name}`);
        await supabase.storage.from('analysis-images').remove(paths);
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
          else if (fnError.message && fnError.message !== 'Edge Function returned a non-2xx status code') {
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

  const FeedbackBanner: React.FC<{ msg: { type: 'success' | 'error'; text: string } | null }> = ({ msg }) => {
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
      {/* ── Main ── */}
      <main className="max-w-2xl mx-auto w-full px-4 sm:px-6 py-8 flex flex-col gap-8 pb-16">
        {/* Back + title */}
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="p-2 hover:bg-slate-100 rounded-full transition-colors text-slate-400 hover:text-slate-600"
            aria-label="Back to Dashboard"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">My Profile</h1>
        </div>

        {/* ── Section 1: Profile Info ── */}
        <section className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-8 shadow-sm">
          <div className="flex items-center gap-4 mb-6">
            {/* Initials avatar */}
            <div className="w-16 h-16 rounded-full bg-teal-100 flex items-center justify-center text-teal-700 font-bold text-xl shrink-0 select-none">
              {initials}
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-900">{fullName || 'Unnamed User'}</h2>
              <p className="text-sm text-slate-500">{email}</p>
            </div>
          </div>

          <form onSubmit={handleSaveProfile} className="space-y-1">
            <FeedbackBanner msg={profileMsg} />
            <Input
              label="Full Name"
              type="text"
              placeholder="Enter your full name"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
            />
            <Input
              label="Email"
              type="email"
              value={email}
              disabled
            />
            <p className="text-[11px] text-slate-400 -mt-2 mb-4 px-0.5">
              Email cannot be changed from this screen for security reasons.
            </p>
            <div className="pt-2">
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
                  'Save Profile'
                )}
              </Button>
            </div>
          </form>
        </section>

        {/* ── Section 2: Change Password ── */}
        <section className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-8 shadow-sm">
          <h2 className="text-lg font-bold text-slate-900 mb-1">Change Password</h2>
          <p className="text-sm text-slate-500 mb-6">Update your account password. New password must be at least 12 characters.</p>

          <form onSubmit={handleChangePassword} className="space-y-1">
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
              label="Confirm New Password"
              type="password"
              placeholder="••••••••"
              value={confirmNewPassword}
              onChange={(e) => setConfirmNewPassword(e.target.value)}
              required
            />
            <div className="pt-2">
              <Button type="submit" disabled={passwordLoading}>
                {passwordLoading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Changing…
                  </span>
                ) : (
                  'Change Password'
                )}
              </Button>
            </div>
          </form>
        </section>

        {/* ── Section 3: Delete Account ── */}
        <section className="bg-white rounded-3xl border border-red-100 p-6 sm:p-8 shadow-sm">
          <h2 className="text-lg font-bold text-red-700 mb-1">Delete Account</h2>
          <p className="text-sm text-slate-500 mb-4 leading-relaxed">
            Permanently delete your account and all associated analysis data.
            This action <strong className="text-red-600">cannot be undone</strong>.
          </p>

          {!showDeleteConfirm ? (
            <button
              onClick={() => setShowDeleteConfirm(true)}
              className="px-5 py-2.5 rounded-full text-sm font-medium border border-red-200 text-red-600 hover:bg-red-50 transition-colors"
            >
              Delete My Account
            </button>
          ) : (
            <div className="space-y-4">
              <FeedbackBanner msg={deleteMsg} />
              <div className="p-4 bg-red-50 border border-red-100 rounded-xl">
                <p className="text-sm text-red-700 mb-3 font-medium">
                  Type <span className="font-mono font-bold">DELETE</span> to confirm:
                </p>
                <input
                  type="text"
                  value={deleteConfirmText}
                  onChange={(e) => setDeleteConfirmText(e.target.value)}
                  placeholder="Type DELETE"
                  className="w-full px-4 py-2.5 rounded-lg border border-red-200 bg-white focus:outline-none focus:ring-2 focus:ring-red-500/20 focus:border-red-500 transition-all text-slate-800 placeholder:text-slate-400 text-sm"
                  autoComplete="off"
                />
              </div>
              <div className="flex gap-3">
                <button
                  onClick={handleDeleteAccount}
                  disabled={deleteConfirmText !== 'DELETE' || deleteLoading}
                  className="flex-1 py-2.5 px-4 rounded-full font-medium text-sm bg-red-600 text-white hover:bg-red-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {deleteLoading ? (
                    <span className="flex items-center justify-center gap-2">
                      <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Deleting…
                    </span>
                  ) : (
                    'Permanently Delete'
                  )}
                </button>
                <button
                  onClick={() => {
                    setShowDeleteConfirm(false);
                    setDeleteConfirmText('');
                    setDeleteMsg(null);
                  }}
                  disabled={deleteLoading}
                  className="flex-1 py-2.5 px-4 rounded-full font-medium text-sm border border-slate-200 text-slate-600 hover:bg-slate-50 transition-colors disabled:opacity-50"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </section>
      </main>

      <footer className="py-8 text-center bg-slate-50 mt-auto">
        <p className="text-[11px] font-medium text-slate-400 uppercase tracking-widest leading-relaxed px-6">
          Clinical Support & Diagnostic Aid Suite
        </p>
      </footer>
    </div>
  );
};

export default ProfileScreen;
