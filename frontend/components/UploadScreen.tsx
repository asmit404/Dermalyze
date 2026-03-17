
import React, { useState, useRef, useEffect } from 'react';
import Button from './ui/Button';

const MAX_FILE_BYTES = 10 * 1024 * 1024; // 10 MB
const MAX_DIMENSION_PX = 448;            // resize longest edge to ≤ 448 px
const JPEG_QUALITY = 0.85;

/** Compress/resize a data-URL image using canvas. PNGs are kept lossless; others use JPEG. */
function compressImage(dataUrl: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, MAX_DIMENSION_PX / Math.max(img.width, img.height));
      const canvas = document.createElement('canvas');
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      const ctx = canvas.getContext('2d');
      if (!ctx) { resolve(dataUrl); return; }
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const isPng = dataUrl.startsWith('data:image/png');
      resolve(isPng ? canvas.toDataURL('image/png') : canvas.toDataURL('image/jpeg', JPEG_QUALITY));
    };
    img.onerror = () => reject(new Error('Failed to load image for compression.'));
    img.src = dataUrl;
  });
}

interface UploadScreenProps {
  selectedImage: string | null;
  onImageSelect: (img: string | null) => void;
  onBack: () => void;
  onRunClassification: () => void;
  onError: (message?: string) => void;
}

const UploadScreen: React.FC<UploadScreenProps> = ({ 
  selectedImage, 
  onImageSelect, 
  onBack, 
  onRunClassification,
  onError
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Always start with a clean slate — clear any image left over from a previous session
  useEffect(() => { onImageSelect(null); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const validateAndProcessFile = (file: File) => {
    const validTypes = ['image/jpeg', 'image/png'];
    if (!validTypes.includes(file.type)) {
      onError('Unsupported file type. Please upload a JPEG or PNG image.');
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      onError(`File is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum allowed size is 10 MB.`);
      return;
    }

    const reader = new FileReader();
    reader.onloadend = async () => {
      try {
        const compressed = await compressImage(reader.result as string);
        onImageSelect(compressed);
      } catch {
        onError('Failed to process the image. Please try a different file.');
      }
    };
    reader.onerror = () => {
      onError('Could not read the file. Please try again.');
    };
    reader.readAsDataURL(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      validateAndProcessFile(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      validateAndProcessFile(file);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const clearSelection = (e: React.MouseEvent) => {
    e.stopPropagation();
    onImageSelect(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="flex-1 flex flex-col bg-slate-50">
      <main className="flex-1 flex items-center justify-center p-6 sm:p-12">
        <div className="max-w-xl w-full">
          <div className="bg-white rounded-3xl border border-slate-200 p-8 sm:p-12 shadow-sm">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-slate-900 mb-2 tracking-tight">
                Upload Image for Analysis
              </h2>
              <p className="text-slate-500 text-sm">
                Please provide a clear dermatoscopic image of the lesion.
              </p>
            </div>

            <div 
              className={`relative border-2 border-dashed rounded-2xl transition-all duration-200 flex flex-col items-center justify-center min-h-[300px] p-6 text-center
                ${selectedImage ? 'border-teal-500 bg-teal-50/10' : 'border-slate-200 hover:border-teal-400 hover:bg-slate-50/50'}
                ${isDragging ? 'border-teal-500 bg-teal-50 ring-4 ring-teal-500/10' : ''}
                ${!selectedImage ? 'cursor-pointer' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={selectedImage ? undefined : triggerFileInput}
            >
              <input 
                type="file" 
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/jpeg,image/png"
                className="hidden"
              />

              {!selectedImage ? (
                <>
                  <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center text-slate-400 mb-4">
                    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-semibold text-slate-700">Click to upload or drag and drop</p>
                    <p className="text-xs text-slate-400 uppercase tracking-wider font-medium">Accepted formats: JPG or PNG</p>
                  </div>
                </>
              ) : (
                <div className="w-full h-full flex flex-col items-center">
                  <div className="relative group max-w-full">
                    <img 
                      src={selectedImage} 
                      alt="Dermatoscopic Preview" 
                      className="max-h-[240px] w-auto rounded-lg shadow-md border border-white"
                    />
                    <button 
                      onClick={clearSelection}
                      className="absolute -top-3 -right-3 bg-white text-slate-400 hover:text-red-500 rounded-full p-1.5 shadow-lg border border-slate-100 transition-colors"
                      title="Remove image"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                  <p className="mt-4 text-xs font-medium text-teal-600 bg-teal-50 px-3 py-1 rounded-full">
                    Image selected successfully
                  </p>
                </div>
              )}
            </div>

            <div className="mt-10 flex flex-col gap-3">
              <Button disabled={!selectedImage} onClick={onRunClassification}>
                <div className="flex items-center justify-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Run Classification
                </div>
              </Button>
              <Button variant="secondary" onClick={onBack}>
                Cancel
              </Button>
            </div>
          </div>
        </div>
      </main>

      <footer className="py-8 text-center bg-slate-50">
        <p className="text-[11px] font-medium text-slate-400 uppercase tracking-widest leading-relaxed px-6">
          Designed to assist medical professionals. Not a replacement for clinical diagnosis.
        </p>
      </footer>
    </div>
  );
};

export default UploadScreen;
