import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import EnrollmentPage from './pages/student/EnrollmentPage';
import QuestionnairePage from './pages/student/QuestionnairePage';
import ResultPage from './pages/student/ResultPage';
import LoginPage from './pages/admin/LoginPage';
import SignupPage from './pages/admin/SignupPage';
import DashboardPage from './pages/admin/DashboardPage';
import AdminLayout from './pages/admin/AdminLayout';
import StudentsPage from './pages/admin/StudentsPage';
import ReportsPage from './pages/admin/ReportsPage';
import SettingsPage from './pages/admin/SettingsPage';

export default function App() {
  return (
    <Router>
      <Routes>
        {/* Student Flow */}
        <Route path="/" element={<EnrollmentPage />} />
        <Route path="/questionnaire" element={<QuestionnairePage />} />
        <Route path="/result" element={<ResultPage />} />

        {/* Admin Flow */}
        <Route path="/admin/login" element={<LoginPage />} />
        <Route path="/admin/signup" element={<SignupPage />} />
        <Route path="/admin" element={<AdminLayout />}>
          <Route path="dashboard" element={<DashboardPage />} />
          <Route path="students" element={<StudentsPage />} />
          <Route path="reports" element={<ReportsPage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </Router>
  );
}
