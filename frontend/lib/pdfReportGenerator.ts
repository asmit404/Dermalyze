import { jsPDF } from 'jspdf';
import type { ClassResult } from './types';
import type { InferenceMetadata } from './types';
import type { ClassInfo, RiskSeverity } from './classInfo';
import { getRiskSeverity } from './classInfo';

export interface ReportData {
  caseId: string;
  date: string;
  time?: string;
  clinicianName: string;
  classId: string;
  className: string;
  confidence: number;
  classInfo: ClassInfo;
  allScores?: ClassResult[] | null;
  notes?: string;
  imageDataUrl?: string;
  metadata?: InferenceMetadata | null; // Patient metadata captured at analysis time
}

function getRiskColor(severity: RiskSeverity): [number, number, number] {
  switch (severity) {
    case 'critical':
      return [220, 38, 38]; // red-600
    case 'high':
      return [234, 88, 12]; // orange-600
    case 'moderate':
      return [217, 119, 6]; // amber-600
    default:
      return [5, 150, 105]; // emerald-600
  }
}

const COLORS = {
  primary: [13, 148, 136] as [number, number, number], // teal-600
  text: [26, 26, 26] as [number, number, number], // slate-900
  secondary: [100, 116, 139] as [number, number, number], // slate-500
  border: [226, 232, 240] as [number, number, number], // slate-200
  bg: [248, 250, 252] as [number, number, number], // slate-50
};

function setColor(pdf: jsPDF, color: [number, number, number]) {
  pdf.setTextColor(color[0], color[1], color[2]);
}

function setFillColor(pdf: jsPDF, color: [number, number, number]) {
  pdf.setFillColor(color[0], color[1], color[2]);
}

function setDrawColor(pdf: jsPDF, color: [number, number, number]) {
  pdf.setDrawColor(color[0], color[1], color[2]);
}

export async function generateDermatologyReport(data: ReportData): Promise<void> {
  const pdf = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  });

  const pageWidth = 210;
  const pageHeight = 297;
  const margin = 20;
  const contentWidth = pageWidth - 2 * margin;
  let y = margin;

  const riskSeverity = getRiskSeverity(data.classInfo.riskLevel);
  const riskColor = getRiskColor(riskSeverity);
  const timestamp = new Date().toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  // Helper: Add text with wrapping
  const addText = (text: string, x: number, maxWidth: number, lineHeight: number = 5) => {
    const lines = pdf.splitTextToSize(text, maxWidth);
    pdf.text(lines, x, y);
    y += lines.length * lineHeight;
    return lines.length * lineHeight;
  };

  // === HEADER ===
  // Logo
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(20);
  setColor(pdf, COLORS.primary);
  pdf.text('DERMALYZE', margin, y);

  // Document ID (right-aligned)
  pdf.setFontSize(9);
  setColor(pdf, COLORS.secondary);
  pdf.setFont('helvetica', 'normal');
  const docIdText = `Doc ID: ${data.caseId}`;
  const docIdWidth = pdf.getTextWidth(docIdText);
  pdf.text(docIdText, pageWidth - margin - docIdWidth, y);

  y += 6;

  // Title
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(11);
  setColor(pdf, COLORS.text);
  pdf.text('DERMATOLOGY ANALYSIS REPORT', margin, y);
  y += 4;

  // Subtitle
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(8);
  setColor(pdf, COLORS.secondary);
  pdf.text('AI-Assisted Skin Lesion Classification', margin, y);
  y += 6;

  // Top border
  setDrawColor(pdf, COLORS.primary);
  pdf.setLineWidth(0.8);
  pdf.line(margin, y, pageWidth - margin, y);
  y += 8;

  // === CASE INFORMATION BAR ===
  const boxY = y;
  setFillColor(pdf, COLORS.bg);
  setDrawColor(pdf, COLORS.border);
  pdf.setLineWidth(0.3);
  pdf.rect(margin, boxY, contentWidth, 18, 'FD');

  // Left accent
  setFillColor(pdf, COLORS.primary);
  pdf.rect(margin, boxY, 2, 18, 'F');

  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(7);
  setColor(pdf, COLORS.primary);

  const colWidth = contentWidth / 3;

  // Column 1: Case ID
  pdf.text('CASE ID', margin + 8, boxY + 6);
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(9);
  setColor(pdf, COLORS.text);
  pdf.text(data.caseId, margin + 8, boxY + 12);

  // Column 2: Date
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(7);
  setColor(pdf, COLORS.primary);
  pdf.text('ANALYSIS DATE', margin + 8 + colWidth, boxY + 6);
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(9);
  setColor(pdf, COLORS.text);
  const dateStr = data.time ? `${data.date}, ${data.time}` : data.date;
  pdf.text(dateStr, margin + 8 + colWidth, boxY + 12);

  // Column 3: Clinician
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(7);
  setColor(pdf, COLORS.primary);
  pdf.text('CLINICIAN', margin + 8 + colWidth * 2, boxY + 6);
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(9);
  setColor(pdf, COLORS.text);
  const clinicianLines = pdf.splitTextToSize(data.clinicianName, colWidth - 8);
  pdf.text(clinicianLines, margin + 8 + colWidth * 2, boxY + 12);

  y += 24;

  // === PATIENT INFORMATION (only if metadata was provided) ===
  const meta = data.metadata;
  const hasAge = meta?.ageApprox != null;
  const hasSex = meta?.sex != null && meta.sex.trim() !== '';
  const hasSite = meta?.anatomSite != null && meta.anatomSite.trim() !== '';
  const hasMetadata = hasAge || hasSex || hasSite;

  // Dynamic section counter — starts at I if no metadata, II if metadata exists
  const ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'];
  let sectionIdx = hasMetadata ? 1 : 0; // 0-based index into ROMAN
  const nextSection = () => ROMAN[sectionIdx++];

  if (hasMetadata) {
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(8);
    setColor(pdf, COLORS.text);
    pdf.text('I. PATIENT INFORMATION', margin, y);
    y += 2;
    setDrawColor(pdf, COLORS.primary);
    pdf.setLineWidth(0.5);
    pdf.line(margin, y, margin + 60, y);
    y += 5;

    const patientBoxY = y;
    const patientFields: { label: string; value: string }[] = [];
    if (hasAge) patientFields.push({ label: 'AGE', value: `${meta!.ageApprox} years` });
    if (hasSex) patientFields.push({ label: 'SEX', value: meta!.sex!.charAt(0).toUpperCase() + meta!.sex!.slice(1) });
    if (hasSite) patientFields.push({ label: 'ANATOMICAL SITE', value: meta!.anatomSite!.split('_').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ') });

    const patientBoxHeight = 14;
    setFillColor(pdf, COLORS.bg);
    setDrawColor(pdf, COLORS.border);
    pdf.setLineWidth(0.3);
    pdf.rect(margin, patientBoxY, contentWidth, patientBoxHeight, 'FD');
    setFillColor(pdf, COLORS.primary);
    pdf.rect(margin, patientBoxY, 2, patientBoxHeight, 'F');

    const fieldColWidth = contentWidth / patientFields.length;
    patientFields.forEach((field, idx) => {
      const fx = margin + 8 + idx * fieldColWidth;
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(7);
      setColor(pdf, COLORS.primary);
      pdf.text(field.label, fx, patientBoxY + 5);
      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(9);
      setColor(pdf, COLORS.text);
      pdf.text(field.value, fx, patientBoxY + 11);
    });

    y += patientBoxHeight + 8;
  }

  // === PRIMARY DIAGNOSIS ===
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(8);
  setColor(pdf, COLORS.text);
  pdf.text(`${nextSection()}. PRIMARY DIAGNOSIS`, margin, y);
  y += 2;

  setDrawColor(pdf, COLORS.primary);
  pdf.setLineWidth(0.5);
  pdf.line(margin, y, margin + 60, y);
  y += 6;

  // Diagnosis box
  const diagBoxY = y;
  setDrawColor(pdf, COLORS.primary);
  pdf.setLineWidth(0.8);
  pdf.rect(margin, diagBoxY, contentWidth, 18);

  // Diagnosis name
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(16);
  setColor(pdf, COLORS.text);
  pdf.text(data.className, margin + 6, diagBoxY + 8);

  // Class ID badge
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(8);
  setColor(pdf, COLORS.primary);
  setDrawColor(pdf, COLORS.primary);
  pdf.setLineWidth(0.4);
  const badgeText = data.classId.toUpperCase();
  const badgeWidth = pdf.getTextWidth(badgeText) + 8;
  pdf.rect(margin + 6, diagBoxY + 11, badgeWidth, 5);
  pdf.text(badgeText, margin + 10, diagBoxY + 14.5);

  // Risk badge (next to class ID badge)
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(7);
  setFillColor(pdf, riskColor);
  const riskText = data.classInfo.riskLevel.toUpperCase();
  const riskWidth = pdf.getTextWidth(riskText) + 6;
  const riskBadgeX = margin + 6 + badgeWidth + 3;
  pdf.roundedRect(riskBadgeX, diagBoxY + 11, riskWidth, 5, 1, 1, 'F');
  pdf.setTextColor(255, 255, 255);
  pdf.text(riskText, riskBadgeX + 3, diagBoxY + 14.5);

  // Confidence (right side)
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(7);
  setColor(pdf, COLORS.secondary);
  const confLabel = 'CONFIDENCE';
  const confLabelWidth = pdf.getTextWidth(confLabel);
  pdf.text(confLabel, pageWidth - margin - 35, diagBoxY + 5);

  pdf.setFontSize(20);
  setColor(pdf, COLORS.primary);
  const confValue = `${data.confidence.toFixed(1)}%`;
  const confValueWidth = pdf.getTextWidth(confValue);
  pdf.text(confValue, pageWidth - margin - confValueWidth / 2 - 17.5, diagBoxY + 12);

  y = diagBoxY + 28;

  // === TWO COLUMN SECTION: IMAGE & DETAILS ===
  const col1X = margin;
  const col2X = margin + contentWidth / 2 + 3;
  const colW = contentWidth / 2 - 3;

  // Column 1: Image
  const img1Y = y;
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(8);
  setColor(pdf, COLORS.text);
  pdf.text(`${nextSection()}. ANALYZED IMAGE`, col1X, img1Y);
  const imgBoxY = img1Y + 5;

  setDrawColor(pdf, COLORS.border);
  pdf.setLineWidth(0.3);
  setFillColor(pdf, COLORS.bg);
  pdf.rect(col1X, imgBoxY, colW, colW * 0.75, 'FD');

  if (data.imageDataUrl) {
    try {
      // Detect image format from data URL
      const imageFormat = data.imageDataUrl.startsWith('data:image/png') ? 'PNG' : 'JPEG';
      pdf.addImage(
        data.imageDataUrl,
        imageFormat,
        col1X + 2,
        imgBoxY + 2,
        colW - 4,
        colW * 0.75 - 4,
        undefined,
        'FAST'
      );
    } catch {
      // Image failed - leave empty box
    }
  }

  // Column 2: Clinical Details
  const det1Y = y;
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(8);
  setColor(pdf, COLORS.text);
  pdf.text(`${nextSection()}. CLINICAL CHARACTERISTICS`, col2X, det1Y);

  let detailY = det1Y + 5;

  const detailBoxHeight = colW * 0.6; // Thinner than image box
  setDrawColor(pdf, COLORS.border);
  pdf.setLineWidth(0.3);
  pdf.rect(col2X, detailY, colW, detailBoxHeight);

  detailY += 4;

  // Description
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(7);
  setColor(pdf, COLORS.primary);
  pdf.text('DESCRIPTION', col2X + 3, detailY);
  detailY += 4;
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(8);
  setColor(pdf, COLORS.text);
  const descLines = pdf.splitTextToSize(data.classInfo.description, colW - 6);
  pdf.text(descLines, col2X + 3, detailY);
  detailY += descLines.length * 4 + 3;

  // Separator
  setDrawColor(pdf, [241, 245, 249]);
  pdf.setLineWidth(0.2);
  pdf.line(col2X + 3, detailY, col2X + colW - 3, detailY);
  detailY += 3;

  // Key Features
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(7);
  setColor(pdf, COLORS.primary);
  pdf.text('KEY CLINICAL FEATURES', col2X + 3, detailY);
  detailY += 4;
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(8);
  setColor(pdf, COLORS.text);
  const featLines = pdf.splitTextToSize(data.classInfo.keyFeatures, colW - 6);
  pdf.text(featLines, col2X + 3, detailY);
  detailY += featLines.length * 4 + 3;

  // Separator
  pdf.line(col2X + 3, detailY, col2X + colW - 3, detailY);
  detailY += 3;

  // Demographics
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(7);
  setColor(pdf, COLORS.primary);
  pdf.text('COMMON DEMOGRAPHICS', col2X + 3, detailY);
  detailY += 4;
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(8);
  setColor(pdf, COLORS.text);
  const demoLines = pdf.splitTextToSize(data.classInfo.commonIn, colW - 6);
  pdf.text(demoLines, col2X + 3, detailY);

  y = imgBoxY + colW * 0.75 + 6;

  // === DIFFERENTIAL DIAGNOSIS ===
  if (data.allScores && data.allScores.length > 0) {
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(8);
    setColor(pdf, COLORS.text);
    pdf.text(`${nextSection()}. DIFFERENTIAL DIAGNOSIS PROBABILITIES`, margin, y);
    y += 2;
    setDrawColor(pdf, COLORS.primary);
    pdf.setLineWidth(0.5);
    pdf.line(margin, y, margin + 80, y);
    y += 4;

    const sortedScores = [...data.allScores].sort((a, b) => b.score - a.score);

    setDrawColor(pdf, COLORS.border);
    pdf.setLineWidth(0.3);
    const rowHeight = 6;
    pdf.rect(margin, y, contentWidth, sortedScores.length * rowHeight + 4);

    y += 3;

    sortedScores.forEach((score) => {
      const isPredicted = score.id === data.classId;

      // Label
      pdf.setFont('helvetica', isPredicted ? 'bold' : 'normal');
      pdf.setFontSize(8);
      setColor(pdf, isPredicted ? COLORS.text : COLORS.secondary);
      pdf.text(score.name, margin + 3, y + 3.5);

      // Bar
      const barX = margin + 55;
      const barWidth = contentWidth - 70;
      const barHeight = 3.5;

      setFillColor(pdf, [241, 245, 249]);
      pdf.rect(barX, y + 0.5, barWidth, barHeight, 'F');

      const fillWidth = (score.score / 100) * barWidth;
      if (isPredicted) {
        setFillColor(pdf, COLORS.primary);
      } else {
        setFillColor(pdf, [203, 213, 225]);
      }
      pdf.rect(barX, y + 0.5, fillWidth, barHeight, 'F');

      // Value
      pdf.setFont('helvetica', isPredicted ? 'bold' : 'normal');
      pdf.setFontSize(8);
      setColor(pdf, isPredicted ? COLORS.primary : COLORS.secondary);
      const valueText = `${score.score.toFixed(1)}%`;
      const valueWidth = pdf.getTextWidth(valueText);
      pdf.text(valueText, pageWidth - margin - valueWidth - 3, y + 3.5);

      y += rowHeight;
    });

    y += 5;
  }

  // === CLINICAL NOTES ===
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(8);
  setColor(pdf, COLORS.text);
  pdf.text(`${nextSection()}. CLINICAL NOTES & OBSERVATIONS`, margin, y);
  y += 2;
  setDrawColor(pdf, COLORS.primary);
  pdf.setLineWidth(0.5);
  pdf.line(margin, y, margin + 70, y);
  y += 5;

  const notesBoxY = y;
  const notesBoxHeight = 18;
  setDrawColor(pdf, COLORS.border);
  pdf.setLineWidth(0.3);
  setFillColor(pdf, [255, 254, 247]);
  pdf.rect(margin, notesBoxY, contentWidth, notesBoxHeight, 'FD');

  // Left accent
  setFillColor(pdf, [251, 191, 36]);
  pdf.rect(margin, notesBoxY, 2, notesBoxHeight, 'F');

  y += 4;
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(8);
  setColor(pdf, COLORS.text);

  if (data.notes && data.notes.trim()) {
    const notesLines = pdf.splitTextToSize(data.notes, contentWidth - 8);
    pdf.text(notesLines.slice(0, 3), margin + 4, y); // Max 3 lines to fit
  } else {
    pdf.setFont('helvetica', 'italic');
    setColor(pdf, COLORS.secondary);
    pdf.text('No clinical notes recorded for this analysis.', margin + 4, y);
  }

  y = notesBoxY + notesBoxHeight + 8;

  // === FOOTER ===

  setDrawColor(pdf, COLORS.primary);
  pdf.setLineWidth(0.8);
  pdf.line(margin, y, pageWidth - margin, y);
  y += 5;

  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(7);
  setColor(pdf, COLORS.primary);
  pdf.text('WARNING: IMPORTANT MEDICAL DISCLAIMER', margin, y);
  y += 4;

  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(7);
  setColor(pdf, COLORS.secondary);
  const disclaimer =
    'This report is generated by an AI-assisted diagnostic tool for educational and screening purposes only. Classification results are probabilistic outputs and do not constitute a definitive medical diagnosis. All findings must be correlated with comprehensive clinical examination, patient history, and histopathological analysis when indicated. Consultation with a board-certified dermatologist is recommended.';
  const disclaimerLines = pdf.splitTextToSize(disclaimer, contentWidth);
  pdf.text(disclaimerLines, margin, y);
  y += disclaimerLines.length * 2.8 + 3;

  setDrawColor(pdf, COLORS.border);
  pdf.setLineWidth(0.2);
  pdf.line(margin, y, pageWidth - margin, y);
  y += 3;

  pdf.setFontSize(6);
  setColor(pdf, [148, 163, 184]);
  pdf.text(`Report Generated: ${timestamp}`, margin, y);

  pdf.setFont('helvetica', 'bold');
  setColor(pdf, COLORS.primary);
  const poweredText = 'POWERED BY DERMALYZE AI';
  const poweredWidth = pdf.getTextWidth(poweredText);
  pdf.text(poweredText, pageWidth - margin - poweredWidth, y);

  // Save
  const filename = `Dermalyze_Report_${data.caseId.replace(/[^a-zA-Z0-9]/g, '_')}.pdf`;
  pdf.save(filename);
}
