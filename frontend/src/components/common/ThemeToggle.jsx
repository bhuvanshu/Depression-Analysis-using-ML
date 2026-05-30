import { useTheme } from '../../context/ThemeContext';
import { Sun, Moon } from 'lucide-react';
import './ThemeToggle.css';

export default function ThemeToggle({ floating = false }) {
  const { theme, toggleTheme } = useTheme();

  return (
    <button 
      className={`theme-toggle ${floating ? 'theme-toggle-floating' : 'theme-toggle-inline'}`} 
      onClick={toggleTheme}
      type="button"
      aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
    >
      <div className="theme-toggle-track">
        <div className={`theme-toggle-thumb ${theme === 'light' ? 'is-light' : 'is-dark'}`} />
        <div className="theme-toggle-icon-wrapper">
          <Sun className="theme-toggle-icon sun" size={14} />
        </div>
        <div className="theme-toggle-icon-wrapper">
          <Moon className="theme-toggle-icon moon" size={14} />
        </div>
      </div>
    </button>
  );
}
