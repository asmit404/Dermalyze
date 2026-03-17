/**
 * Maps raw Supabase auth error messages to user-friendly strings.
 * Prevents internal implementation details from surfacing in the UI.
 */
const AUTH_ERROR_MAP: Record<string, string> = {
  'Invalid login credentials':                       'Invalid email or password.',
  'Email not confirmed':                             'Please verify your email before logging in. Check your inbox.',
  'User not found':                                  'No account found with that email address.',
  'Password should be at least 6 characters':        'Password must be at least 12 characters.',
  'New password should be different from the old password':
                                                     'New password must differ from the current password.',
  'Auth session missing':                            'Your session has expired. Please log in again.',
  'Token has expired or is invalid':                 'This link has expired. Please request a new one.',
  'Email rate limit exceeded':                       'Too many requests. Please wait a moment before trying again.',
  'over_email_send_rate_limit':                      'Too many requests. Please wait a moment before trying again.',
};

export function friendlyAuthError(message: string): string {
  for (const [fragment, friendly] of Object.entries(AUTH_ERROR_MAP)) {
    if (message.includes(fragment)) return friendly;
  }
  return 'An unexpected error occurred. Please try again.';
}
