
import React from 'react';
import Button from './ui/Button';

interface ErrorScreenProps {
  onBackToUpload: () => void;
  onRetry?: () => void;
  message?: string;
}

const ErrorScreen: React.FC<ErrorScreenProps> = ({ onBackToUpload, onRetry, message }) => {
  return (
    <div className="flex-1 flex flex-col bg-slate-50">
      {/* Main Error State */}
      <main className="flex-1 flex items-center justify-center p-6 sm:p-12">
        <div className="max-w-md w-full text-center">
          <div className="bg-white rounded-3xl border border-slate-200 p-12 sm:p-16 shadow-sm flex flex-col items-center">
            
            {/* Calm Warning Icon */}
            <div className="w-20 h-20 bg-amber-50 rounded-full flex items-center justify-center text-amber-500 mb-8">
              <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>

            <h2 className="text-2xl font-bold text-slate-900 mb-3 tracking-tight">
              {message ? 'Analysis Error' : 'Invalid Image Format'}
            </h2>
            <p className="text-slate-500 text-sm mb-10 leading-relaxed px-4">
              {message ?? 'Please upload a valid JPG or PNG image. The system requires standard image formats for diagnostic analysis.'}
            </p>

            <div className="w-full space-y-3">
              {onRetry && (
                <Button onClick={onRetry}>
                  <div className="flex items-center justify-center gap-2">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Try Again
                  </div>
                </Button>
              )}
              <Button onClick={onBackToUpload} variant={onRetry ? 'secondary' : 'primary'}>
                <div className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                  Upload Another Image
                </div>
              </Button>
              <button 
                onClick={onBackToUpload}
                className="text-xs font-bold text-slate-400 uppercase tracking-widest hover:text-slate-600 transition-colors py-2"
              >
                Go Back
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Footer Branding */}
      <footer className="py-8 text-center bg-slate-50">
        <p className="text-[11px] font-medium text-slate-400 uppercase tracking-widest leading-relaxed px-6">
          Designed to assist medical professionals. Not a replacement for clinical diagnosis.
        </p>
      </footer>
    </div>
  );
};

export default ErrorScreen;
