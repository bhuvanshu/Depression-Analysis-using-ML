import { useState } from 'react';
import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { Brain, LayoutDashboard, Users, FileBarChart, Settings, LogOut, Menu, X } from 'lucide-react';
import './AdminLayout.css';

export default function AdminLayout() {
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const admin = JSON.parse(localStorage.getItem('admin_auth') || '{}');

  const handleLogout = () => {
    localStorage.removeItem('admin_auth');
    navigate('/admin/login');
  };

  const getInitials = (name) => name ? name.split(' ').map(n => n[0]).join('').toUpperCase() : '?';

  const navItems = [
    { path: '/admin/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/admin/students', label: 'Students', icon: Users },
    { path: '/admin/reports', label: 'Reports', icon: FileBarChart },
    { path: '/admin/settings', label: 'Settings', icon: Settings }
  ];

  return (
    <div className="admin-layout">
      {/* Mobile Toggle */}
      <button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
        {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="sidebar-logo-icon">
              <Brain size={20} color="white" />
            </div>
            <div>
              <div className="sidebar-logo-text">Mind Care</div>
              <div className="sidebar-logo-sub">Admin Panel</div>
            </div>
          </div>
        </div>

        <nav className="sidebar-nav">
          {navItems.map(item => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `sidebar-nav-item ${isActive ? 'active' : ''}`}
              onClick={() => setSidebarOpen(false)} // Close sidebar on mobile after click
            >
              <item.icon size={18} />
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-user">
            <div className="sidebar-user-avatar">
              {getInitials(admin.name || 'Admin')}
            </div>
            <div className="sidebar-user-info">
              <div className="sidebar-user-name">{admin.name || 'Admin'}</div>
              <div className="sidebar-user-role">{admin.college || 'Institution'}</div>
            </div>
            <button
              onClick={handleLogout}
              style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 4 }}
              title="Logout"
            >
              <LogOut size={16} />
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="admin-main">
        <Outlet />
      </main>
    </div>
  );
}
