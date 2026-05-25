const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080/api';

/**
 * Reads the logged-in admin's college name from localStorage
 * and returns headers that must accompany every API request.
 */
function getAuthHeaders() {
  const headers = {};
  try {
    const auth = JSON.parse(localStorage.getItem('admin_auth'));
    if (auth?.college) {
      headers['X-College-Name'] = auth.college;
    }
  } catch (_) {
    // ignore parse errors
  }
  return headers;
}

/**
 * Helper function for fetch calls
 */
async function apiCall(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;

  const headers = {
    'Content-Type': 'application/json',
    ...getAuthHeaders(),
    ...options.headers,
  };

  const config = {
    ...options,
    headers,
  };

  try {
    const response = await fetch(url, config);
    const contentType = response.headers.get('content-type');

    // For 200 OK responses that are plain text (like signup success message)
    if (response.ok && contentType && !contentType.includes('application/json')) {
      const text = await response.text();
      return text;
    }

    let data = null;
    let textError = null;

    if (contentType && contentType.includes('application/json')) {
      data = await response.json();
    } else {
      textError = await response.text();
    }

    if (!response.ok) {
      const errorMessage = data?.message || data || textError || response.statusText || 'API Error';
      throw new Error(typeof errorMessage === 'string' ? errorMessage : JSON.stringify(errorMessage));
    }

    return data !== null ? data : textError;
  } catch (error) {
    console.error(`API Error (${endpoint}):`, error);
    throw error;
  }
}

// ==============================================
// STUDENT FLOW
// ==============================================

export const verifyStudent = (enrollmentId) =>
  apiCall('/student/verify', {
    method: 'POST',
    body: JSON.stringify({ enrollmentId }),
  });

export const submitScreening = (screeningData) =>
  apiCall('/screening/submit', {
    method: 'POST',
    body: JSON.stringify(screeningData),
  });

// ==============================================
// ADMIN FLOW
// ==============================================

export const adminSignup = (signupData) =>
  apiCall('/admin/signup', {
    method: 'POST',
    body: JSON.stringify(signupData),
  });

export const adminLogin = (loginData) =>
  apiCall('/admin/login', {
    method: 'POST',
    body: JSON.stringify(loginData),
  });

// ==============================================
// DASHBOARD & REPORTS
// ==============================================

export const getDashboardSummary = () =>
  apiCall('/dashboard/summary', { method: 'GET' });

export const getDashboardStudents = () =>
  apiCall('/dashboard/students', { method: 'GET' });

export const getHighRiskStudents = () =>
  apiCall('/dashboard/high-risk', { method: 'GET' });

// ==============================================
// STUDENT MANAGEMENT
// ==============================================

export const addStudent = (studentData) =>
  apiCall('/student/add', {
    method: 'POST',
    body: JSON.stringify(studentData),
  });

export const bulkAddStudents = (students) =>
  apiCall('/student/bulk', {
    method: 'POST',
    body: JSON.stringify(students),
  });

