import { afterEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import RecruiterPortal from './RecruiterPortal.jsx'

const recruiter = { id: 10, full_name: 'Riley Recruiter', role: 'recruiter' }
const job = {
  id: 7,
  title: 'Data Engineer',
  company_name: 'Example Co',
  status: 'active',
  location: 'Remote',
  experience_required: 4,
}

function response(body, status = 200) {
  return { ok: status < 400, status, json: async () => body }
}

function makeApi({ authenticated = true, jobs = [job], applicants = [], loginResponse, uploadResponse } = {}) {
  const state = { authenticated, jobs: [...jobs], applicants: [...applicants], shortlist: [] }
  const fetchMock = vi.fn(async (url, options = {}) => {
    const method = options.method || 'GET'
    if (url === '/api/auth/me') {
      return state.authenticated
        ? response(recruiter)
        : response({ detail: 'Recruiter authentication is required.' }, 401)
    }
    if (url === '/api/auth/login' && method === 'POST') {
      const result = loginResponse || response(recruiter)
      if (result.ok) state.authenticated = true
      return result
    }
    if (url === '/api/jobs/' && method === 'GET') return response(state.jobs)
    if (url === '/api/jobs/' && method === 'POST') {
      const created = { ...job, id: 8, title: 'Platform Engineer' }
      state.jobs = [...state.jobs, created]
      return response(created, 201)
    }
    if (url === '/api/applicants/' && method === 'GET') return response(state.applicants)
    if (url === '/api/applicants/jobs/7' && method === 'GET') return response(state.applicants)
    if (url === '/api/applicants/jobs/7/screening-runs' && method === 'GET') return response([])
    if (url === '/api/applicants/jobs/7/shortlist' && method === 'GET') return response(state.shortlist)
    if (url === '/api/applicants/bulk-upload?job_id=7' && method === 'POST') {
      return uploadResponse || response({
        job_id: 7,
        total_files: 1,
        completed: 1,
        failed: 0,
        results: [{ filename: 'alex.pdf', status: 'completed', role: 'Data Engineer', years_experience: 6 }],
      }, 201)
    }
    if (url === '/api/applicants/jobs/7/rank' && method === 'POST') return response({
      job_id: 7,
      job_title: 'Data Engineer',
      screening_run_id: 33,
      ranked_applicant_count: 1,
      skipped_applicant_count: 0,
      results: [{
        applicant_id: 21,
        filename: 'alex.pdf',
        role: 'Data Engineer',
        years_experience: 6,
        seniority: 'senior',
        education: 'M.S. Computer Science',
        rank: 1,
        ranking_score: 0.93,
        skill_coverage: 0.75,
        matched_skills: ['Python'],
        missing_skills: ['Spark'],
        matched_preferred_skills: ['AWS'],
        missing_preferred_skills: [],
        role_compatibility: 1,
        experience_compatibility: 0.8,
        seniority_compatibility: 0.9,
        education_compatibility: 1,
      }],
      skipped_applicants: [],
    })
    if (url === '/api/applicants/jobs/7/applicants/21/shortlist' && method === 'POST') {
      state.shortlist = [{
        id: 90,
        applicant_id: 21,
        filename: 'alex.pdf',
        role: 'Data Engineer',
        years_experience: 6,
        seniority: 'senior',
        education: 'M.S. Computer Science',
      }]
      return response(state.shortlist[0], 201)
    }
    return response({ detail: `Unexpected request: ${method} ${url}` }, 500)
  })
  vi.stubGlobal('fetch', fetchMock)
  return { fetchMock, state }
}

async function signIn(user) {
  await user.type(screen.getByLabelText('Email'), 'riley@example.com')
  await user.type(screen.getByLabelText('Password'), 'safe-test-password')
  await user.click(screen.getByRole('button', { name: 'Sign in' }))
  await screen.findByRole('button', { name: 'Log out' })
}

async function openScreening(user) {
  await user.click(screen.getByRole('button', { name: 'Applicant Screening' }))
  await user.selectOptions(screen.getByLabelText('Active job'), '7')
  await screen.findByText('No shortlisted applicants')
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('recruiter portal workflows', () => {
  it('shows login errors and returns to an enabled login form', async () => {
    makeApi({
      authenticated: false,
      loginResponse: response({ detail: 'Invalid email or password.' }, 401),
    })
    const user = userEvent.setup()
    render(<RecruiterPortal />)

    await screen.findByRole('heading', { name: 'Welcome back' })
    await user.type(screen.getByLabelText('Email'), 'riley@example.com')
    await user.type(screen.getByLabelText('Password'), 'wrong-password')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Invalid email or password.')
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeEnabled()
  })

  it('signs in, lists jobs, and creates a job with the entered details', async () => {
    const { fetchMock } = makeApi({ authenticated: false })
    const user = userEvent.setup()
    render(<RecruiterPortal />)
    await screen.findByRole('heading', { name: 'Welcome back' })
    await signIn(user)

    await user.click(screen.getByRole('button', { name: 'Jobs' }))
    expect(await screen.findByText('Data Engineer')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Create a job' }))
    const dialog = screen.getByRole('dialog')
    await user.type(within(dialog).getByLabelText('Job title'), 'Platform Engineer')
    await user.type(within(dialog).getByLabelText('Company'), 'Example Co')
    await user.type(within(dialog).getByLabelText('Job description'), 'Build dependable data services.')
    await user.click(within(dialog).getByRole('button', { name: 'Create and process job' }))

    expect(await screen.findByRole('status')).toHaveTextContent('Job created and processed.')
    expect(await screen.findByText('Platform Engineer')).toBeInTheDocument()
    const createCall = fetchMock.mock.calls.find(([url, options]) => url === '/api/jobs/' && options?.method === 'POST')
    expect(createCall).toBeTruthy()
    expect(JSON.parse(createCall[1].body)).toMatchObject({
      title: 'Platform Engineer',
      company_name: 'Example Co',
      description: 'Build dependable data services.',
    })
  })

  it('shows upload loading and success states, then refreshes the applicant pool', async () => {
    let resolveUpload
    const { fetchMock, state } = makeApi()
    fetchMock.mockImplementation(async (url, options = {}) => {
      if (url === '/api/applicants/bulk-upload?job_id=7' && options.method === 'POST') {
        return new Promise(resolve => { resolveUpload = resolve })
      }
      return makeApiResponseForCurrentState(url, options, state)
    })
    const user = userEvent.setup()
    render(<RecruiterPortal />)
    await screen.findByRole('button', { name: 'Log out' })
    await openScreening(user)
    const file = new File(['resume text'], 'alex.pdf', { type: 'application/pdf' })
    fireEvent.change(screen.getByLabelText(/Select multiple resumes/), { target: { files: [file] } })
    const submit = screen.getByRole('button', { name: 'Upload and process batch' })
    await user.click(submit)
    expect(await screen.findByRole('button', { name: 'Uploading and processing...' })).toBeDisabled()

    resolveUpload(response({
      job_id: 7,
      total_files: 1,
      completed: 1,
      failed: 0,
      results: [{ filename: 'alex.pdf', status: 'completed', role: 'Data Engineer', years_experience: 6 }],
    }, 201))
    state.applicants = [{ id: 21, job_id: 7, filename: 'alex.pdf', processing_status: 'completed' }]
    expect(await screen.findByText(/1 completed/)).toBeInTheDocument()
    expect(await screen.findAllByText('alex.pdf')).toHaveLength(2)
    expect(await screen.findByText(/6 years/)).toBeInTheDocument()
  })

  it('shows a readable upload API error and re-enables the upload control', async () => {
    makeApi({ uploadResponse: response({ detail: 'Upload service unavailable.' }, 503) })
    const user = userEvent.setup()
    render(<RecruiterPortal />)
    await screen.findByRole('button', { name: 'Log out' })
    await openScreening(user)
    fireEvent.change(screen.getByLabelText(/Select multiple resumes/), {
      target: { files: [new File(['resume'], 'alex.pdf', { type: 'application/pdf' })] },
    })
    await user.click(screen.getByRole('button', { name: 'Upload and process batch' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Upload service unavailable.')
    expect(screen.getByRole('button', { name: 'Upload and process batch' })).toBeEnabled()
  })

  it('shows ranked candidate score and match evidence, then confirms shortlisting', async () => {
    const { fetchMock } = makeApi({
      applicants: [{ id: 21, job_id: 7, filename: 'alex.pdf', processing_status: 'completed' }],
    })
    const user = userEvent.setup()
    render(<RecruiterPortal />)
    await screen.findByRole('button', { name: 'Log out' })
    await openScreening(user)
    await user.click(screen.getByRole('button', { name: 'Rank applicants' }))

    const candidate = await screen.findByRole('article')
    expect(within(candidate).getByText('alex.pdf')).toBeInTheDocument()
    expect(within(candidate).getByText('0.930')).toBeInTheDocument()
    expect(within(candidate).getByText('Matched: Python')).toBeInTheDocument()
    expect(within(candidate).getByText('Missing: Spark')).toBeInTheDocument()
    expect(within(candidate).getByText('Required skill coverage')).toBeInTheDocument()
    expect(within(candidate).getByText('Role compatibility')).toBeInTheDocument()

    await user.click(within(candidate).getByRole('button', { name: 'Add to shortlist' }))
    expect(await screen.findByRole('status')).toHaveTextContent('Applicant added to this job')
    expect(await screen.findByText('Shortlist')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Shortlisted' })).toBeDisabled()
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/applicants/jobs/7/applicants/21/shortlist',
      expect.objectContaining({ method: 'POST' }),
    )
  })
})

function makeApiResponseForCurrentState(url, options, state) {
  const method = options.method || 'GET'
  if (url === '/api/auth/me') return response(recruiter)
  if (url === '/api/jobs/' && method === 'GET') return response(state.jobs)
  if (url === '/api/applicants/' && method === 'GET') return response(state.applicants)
  if (url === '/api/applicants/jobs/7' && method === 'GET') return response(state.applicants)
  if (url === '/api/applicants/jobs/7/screening-runs' && method === 'GET') return response([])
  if (url === '/api/applicants/jobs/7/shortlist' && method === 'GET') return response(state.shortlist)
  return response({ detail: 'Unexpected request.' }, 500)
}
