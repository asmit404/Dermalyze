import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  BeakerIcon,
  ClockIcon,
  UserCircleIcon,
  InformationCircleIcon,
  QuestionMarkCircleIcon,
  ArrowRightOnRectangleIcon,
  ShieldCheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  Bars3Icon,
} from '@heroicons/react/24/outline';
import { supabase } from '../lib/supabase';
import { ROUTES } from '../App';

const STORAGE_KEY = 'dermalyze_sidebar_expanded';

interface SidebarProps {
  onLogout: () => void;
  /** Mobile overlay open state — controlled by AppLayout */
  mobileOpen: boolean;
  onMobileClose: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ onLogout, mobileOpen, onMobileClose }) => {
  const navigate   = useNavigate();
  const location   = useLocation();

  const [expanded, setExpanded] = useState<boolean>(true);
  const [userId,      setUserId]      = useState('');
  const [userEmail,   setUserEmail]   = useState('');
  const [userName,    setUserName]    = useState('');

  // Persist collapse preference (scoped to user to avoid leaking UX state between accounts)
  useEffect(() => {
    if (!userId) return;
    localStorage.setItem(`${STORAGE_KEY}_${userId}`, String(expanded));
  }, [expanded, userId]);

  // Load user identity and their sidebar preference
  useEffect(() => {
    supabase.auth.getUser().then(({ data: { user } }) => {
      if (user) {
        setUserEmail(user.email ?? '');
        setUserName(user.user_metadata?.full_name ?? '');
        setUserId(user.id);
        const stored = localStorage.getItem(`${STORAGE_KEY}_${user.id}`);
        if (stored !== null) {
          setExpanded(stored !== 'false');
        }
      }
    });
  }, []);

  // ── Helpers ──────────────────────────────────────────────────────────────
  const isActive = (paths: string[]) =>
    paths.some((p) => location.pathname === p || location.pathname.startsWith(p + '/'));

  const go = (path: string) => {
    navigate(path);
    onMobileClose();
  };

  const initials = userName
    ? userName.split(' ').map((n) => n[0]).join('').toUpperCase().slice(0, 2)
    : userEmail ? userEmail[0].toUpperCase() : '?';

  // ── NavItem ───────────────────────────────────────────────────────────────
  interface NavItemProps {
    icon: React.ReactNode;
    label: string;
    active?: boolean;
    onClick: () => void;
    danger?: boolean;
  }

  const NavItem: React.FC<NavItemProps> = ({ icon, label, active, onClick, danger }) => (
    <div className="relative group/nav">
      <button
        onClick={onClick}
        className={[
          'w-full flex items-center gap-3 py-2.5 rounded-lg text-left transition-colors duration-150',
          expanded ? 'px-3' : 'px-0 justify-center',
          active
            ? 'bg-teal-50 text-teal-700'
            : danger
              ? 'text-slate-500 hover:bg-red-50 hover:text-red-600'
              : 'text-slate-500 hover:bg-slate-100 hover:text-slate-700',
          active && expanded ? 'border-l-[3px] border-teal-600 pl-[calc(0.75rem-3px)]' : '',
          active && !expanded ? 'ring-1 ring-teal-200' : '',
        ].join(' ')}
        aria-current={active ? 'page' : undefined}
      >
        <span className="shrink-0 w-5 h-5">{icon}</span>
        {expanded && (
          <span className="flex-1 truncate text-sm font-medium">{label}</span>
        )}
      </button>

      {/* Tooltip — only rendered when sidebar is collapsed */}
      {!expanded && (
        <span
          className="
            absolute left-full top-1/2 -translate-y-1/2 ml-3 z-50
            bg-slate-800 text-white text-xs font-medium px-2.5 py-1.5 rounded-md shadow-lg
            whitespace-nowrap pointer-events-none
            opacity-0 group-hover/nav:opacity-100 transition-opacity duration-150
          "
        >
          {label}
        </span>
      )}
    </div>
  );

  // ── Section label (only when expanded) ──────────────────────────────────
  const SectionLabel: React.FC<{ children: React.ReactNode }> = ({ children }) =>
    expanded ? (
      <p className="px-3 pt-4 pb-1.5 text-[10px] font-bold text-slate-400 uppercase tracking-widest">
        {children}
      </p>
    ) : (
      <div className="my-1 border-t border-slate-100" />
    );

  // ── Sidebar panel ─────────────────────────────────────────────────────────
  return (
    <>
      {/* Mobile backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-30 lg:hidden"
          onClick={onMobileClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={[
          // Base
          'flex flex-col bg-white border-r border-slate-200 shadow-sm',
          'transition-[width,transform] duration-200 ease-in-out',
          // Mobile: fixed overlay; hidden until mobileOpen
          'fixed top-0 left-0 h-screen z-40',
          mobileOpen ? 'translate-x-0' : '-translate-x-full',
          // Desktop: sticky in-flow, always visible
          'lg:sticky lg:top-0 lg:h-screen lg:translate-x-0 lg:z-auto lg:flex-shrink-0',
          // Width
          expanded ? 'w-[220px]' : 'w-[56px]',
        ].join(' ')}
      >
        {/* ── Logo + toggle ── */}
        <div
          className={[
            'flex items-center border-b border-slate-100 h-[57px] px-3',
            expanded ? 'justify-between' : 'justify-center',
          ].join(' ')}
        >
          {expanded ? (
            <>
              <button
                onClick={() => go(ROUTES.dashboard)}
                className="flex items-center gap-2 group/logo"
              >
                <div className="bg-teal-600 rounded-lg p-1.5 group-hover/logo:bg-teal-700 transition-colors">
                  <ShieldCheckIcon className="w-4 h-4 text-white" />
                </div>
                <span className="font-bold text-slate-900 tracking-tight text-base">Dermalyze</span>
              </button>
              <button
                onClick={() => setExpanded(false)}
                className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors"
                aria-label="Collapse sidebar"
              >
                <ChevronLeftIcon className="w-4 h-4" />
              </button>
            </>
          ) : (
            <div className="relative group/logo">
              <button
                onClick={() => setExpanded(true)}
                className="bg-teal-600 rounded-lg p-1.5 hover:bg-teal-700 transition-colors"
                aria-label="Expand sidebar"
              >
                <ChevronRightIcon className="w-4 h-4 text-white" />
              </button>
              <span className="absolute left-full top-1/2 -translate-y-1/2 ml-3 z-50 bg-slate-800 text-white text-xs font-medium px-2.5 py-1.5 rounded-md shadow-lg whitespace-nowrap pointer-events-none opacity-0 group-hover/logo:opacity-100 transition-opacity duration-150">
                Dermalyze
              </span>
            </div>
          )}
        </div>

        {/* ── Navigation ── */}
        <nav className="flex-1 px-2 py-3 space-y-0.5">
          <SectionLabel>Workspace</SectionLabel>

          <NavItem
            icon={<BeakerIcon className="w-5 h-5" />}
            label="New Analysis"
            active={isActive([ROUTES.upload, ROUTES.processing, ROUTES.results])}
            onClick={() => go(ROUTES.upload)}
          />

          <NavItem
            icon={<ClockIcon className="w-5 h-5" />}
            label="History"
            active={isActive([ROUTES.history])}
            onClick={() => go(ROUTES.history)}
          />

          <SectionLabel>Account</SectionLabel>

          <NavItem
            icon={<UserCircleIcon className="w-5 h-5" />}
            label="Profile"
            active={isActive([ROUTES.profile])}
            onClick={() => go(ROUTES.profile)}
          />

          <SectionLabel>Info</SectionLabel>

          <NavItem
            icon={<InformationCircleIcon className="w-5 h-5" />}
            label="About"
            active={location.pathname === ROUTES.about}
            onClick={() => go(ROUTES.about)}
          />
          <NavItem
            icon={<QuestionMarkCircleIcon className="w-5 h-5" />}
            label="Help"
            active={location.pathname === ROUTES.help}
            onClick={() => go(ROUTES.help)}
          />
        </nav>

        {/* ── Bottom: user chip + logout ── */}
        <div className="px-2 py-3 border-t border-slate-100 space-y-0.5">
          {expanded && userEmail && (
            <div className="flex items-center gap-2.5 px-3 py-2 mb-1 rounded-lg bg-slate-50">
              <div className="w-7 h-7 rounded-full bg-teal-100 flex items-center justify-center text-teal-700 text-[11px] font-bold shrink-0">
                {initials}
              </div>
              <div className="flex-1 min-w-0">
                {userName && (
                  <p className="text-xs font-semibold text-slate-700 truncate leading-none mb-0.5">
                    {userName}
                  </p>
                )}
                <p className="text-[11px] text-slate-400 truncate">{userEmail}</p>
              </div>
            </div>
          )}
          <NavItem
            icon={<ArrowRightOnRectangleIcon className="w-5 h-5" />}
            label="Logout"
            onClick={onLogout}
            danger
          />
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
