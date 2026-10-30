import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, AlertCircle, BrainCircuit, BriefcaseBusiness, CheckCircle2,
  ChevronRight, FileText, LogOut, Plus, RefreshCw, Search,
  Upload, Users, X,
} from 'lucide-react'
import './App.css'

const api = async (path, options = {}) => {
  const response = await fetch(path, { credentials: 'same-origin', ...options })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const error = new Error(body.detail || `Request failed (${response.status})`)
    error.status = response.status
    throw error
  }
  return body
}

const fetchDashboardData = async () => {
  const recruiter = await api('/api/auth/me')
  const [jobs, applicants] = await Promise.all([
    api('/api/jobs/'),
    api('/api/applicants/'),
  ])
  return { recruiter, jobs, applicants }
}

const emptyJob = {
  title: '',
  company_name: '',
  location: '',
  industry: '',
  description: '',
  required_skills: '',
  preferred_skills: '',
  experience_required: '',
  seniority_required: '',
  education_required: '',
}

const MAX_FILES = 50
const MAX_FILE_BYTES = 5 * 1024 * 1024

function RecruiterPortal() {
  const [currentUser, setCurrentUser] = useState(null)
  const [authMode, setAuthMode] = useState('login')
  const [authValues, setAuthValues] = useState({ email: '', full_name: '', password: '' })
  const [authBusy, setAuthBusy] = useState(false)
  const [jobs, setJobs] = useState([])
  const [applicants, setApplicants] = useState([])
  const [page, setPage] = useState('Overview')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [search, setSearch] = useState('')
  const [showJobForm, setShowJobForm] = useState(false)
  const [editingJob, setEditingJob] = useState(null)
  const [jobForm, setJobForm] = useState(emptyJob)
  const [screeningJobId, setScreeningJobId] = useState('')
  const [screeningApplicants, setScreeningApplicants] = useState([])
  const [selectedFiles, setSelectedFiles] = useState([])
  const [uploadReport, setUploadReport] = useState(null)
  const [ranking, setRanking] = useState(null)
  const [screeningRuns, setScreeningRuns] = useState([])
  const [shortlistedApplicants, setShortlistedApplicants] = useState([])
  const screeningRequestRef = useRef(0)

  const loadData = useCallback(async () => {
    try {
      const { recruiter, jobs: jobRows, applicants: applicantRows } = await fetchDashboardData()
      setCurrentUser(recruiter)
      setJobs(jobRows)
      setApplicants(applicantRows)
      setError('')
    } catch (err) {
      if (err.status === 401) {
        setCurrentUser(null)
        setJobs([])
        setApplicants([])
      } else {
        setError(`${err.message}. Check the API and database configuration.`)
      }
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    fetchDashboardData().then(({ recruiter, jobs: jobRows, applicants: applicantRows }) => {
      if (!active) return
      setCurrentUser(recruiter)
      setJobs(jobRows)
      setApplicants(applicantRows)
      setError('')
    }).catch(err => {
      if (!active) return
      if (err.status === 401) {
        setCurrentUser(null)
        setJobs([])
        setApplicants([])
      } else {
        setError(`${err.message}. Check the API and database configuration.`)
      }
    }).finally(() => {
      if (active) setLoading(false)
    })
    return () => { active = false }
  }, [])

  const filteredJobs = useMemo(() => jobs.filter(job =>
    `${job.title} ${job.company_name} ${job.location || ''}`
      .toLowerCase().includes(search.trim().toLowerCase())
  ), [jobs, search])
  const jobsWithCounts = useMemo(() => jobs.map(job => ({
    ...job,
    applicant_count: applicants.filter(row => row.job_id === job.id).length,
  })), [jobs, applicants])

  const noticeFor = message => {
    setNotice(message)
    window.setTimeout(() => setNotice(''), 4000)
  }

  const submitAuth = async event => {
    event.preventDefault()
    setAuthBusy(true)
    setError('')
    const registering = authMode === 'register'
    try {
      await api(`/api/auth/${registering ? 'register' : 'login'}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(registering ? authValues : {
          email: authValues.email,
          password: authValues.password,
        }),
      })
      setAuthValues({ email: '', full_name: '', password: '' })
      setLoading(true)
      await loadData()
    } catch (err) {
      setError(err.message)
    } finally {
      setAuthBusy(false)
    }
  }

  const logout = async () => {
    setError('')
    try {
      await api('/api/auth/logout', { method: 'POST' })
    } catch (err) {
      setError(err.message)
    } finally {
      setCurrentUser(null)
      setJobs([])
      setApplicants([])
      setPage('Overview')
    }
  }

  const openNewJob = () => {
    setEditingJob(null)
    setJobForm(emptyJob)
    setShowJobForm(true)
  }

  const openEditJob = job => {
    setEditingJob(job)
    setJobForm({
      title: job.title || '',
      company_name: job.company_name || '',
      location: job.location || '',
      industry: job.industry || '',
      description: job.description || '',
      required_skills: job.required_skills || '',
      preferred_skills: job.preferred_skills || '',
      experience_required: job.experience_required ?? '',
      seniority_required: job.seniority_required || '',
      education_required: job.education_required || '',
    })
    setShowJobForm(true)
  }

  const saveJob = async event => {
    event.preventDefault()
    setBusy(true)
    setError('')
    const payload = {
      ...jobForm,
      experience_required: jobForm.experience_required === ''
        ? null : Number(jobForm.experience_required),
    }
    try {
      await api(editingJob ? `/api/jobs/${editingJob.id}` : '/api/jobs/', {
        method: editingJob ? 'PATCH' : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      setShowJobForm(false)
      noticeFor(editingJob ? 'Job updated.' : 'Job created and processed.')
      await loadData()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const changeJobStatus = async job => {
    const status = job.status === 'active' ? 'closed' : 'active'
    setBusy(true)
    setError('')
    try {
      await api(`/api/jobs/${job.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status }),
      })
      noticeFor(status === 'closed' ? 'Job closed.' : 'Job reopened.')
      await loadData()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const openScreening = async jobId => {
    const requestId = ++screeningRequestRef.current
    setScreeningJobId(String(jobId))
    setUploadReport(null)
    setRanking(null)
    setSelectedFiles([])
    setScreeningApplicants([])
    setScreeningRuns([])
    setShortlistedApplicants([])
    setPage('Applicant Screening')
    setError('')
    if (!jobId) {
      setScreeningApplicants([])
      return
    }
    try {
      const [rows, runs, shortlist] = await Promise.all([
        api(`/api/applicants/jobs/${encodeURIComponent(jobId)}`),
        api(`/api/applicants/jobs/${encodeURIComponent(jobId)}/screening-runs`),
        api(`/api/applicants/jobs/${encodeURIComponent(jobId)}/shortlist`),
      ])
      if (requestId !== screeningRequestRef.current) return
      setScreeningApplicants(rows)
      setScreeningRuns(runs)
      setShortlistedApplicants(shortlist)
    } catch (err) {
      if (requestId !== screeningRequestRef.current) return
      setScreeningApplicants([])
      setError(err.message)
    }
  }

  const chooseFiles = event => {
    const files = Array.from(event.target.files || [])
    setError('')
    if (files.length > MAX_FILES) {
      setSelectedFiles(files.slice(0, MAX_FILES))
      setError(`A batch can contain at most ${MAX_FILES} resumes.`)
    } else {
      setSelectedFiles(files)
    }
    setUploadReport(null)
    setRanking(null)
  }

  const uploadBatch = async event => {
    event.preventDefault()
    const invalid = selectedFiles.find(file => {
      const ext = file.name.split('.').pop()?.toLowerCase()
      return !['pdf', 'docx'].includes(ext) || file.size > MAX_FILE_BYTES
    })
    if (!screeningJobId || selectedFiles.length === 0) {
      setError('Choose an active job and at least one resume.')
      return
    }
    if (invalid) {
      setError(`${invalid.name} must be a PDF or DOCX file no larger than 5 MB.`)
      return
    }

    const jobId = screeningJobId
    const requestId = ++screeningRequestRef.current
    setBusy(true)
    setError('')
    setUploadReport(null)
    setRanking(null)
    try {
      const formData = new FormData()
      selectedFiles.forEach(file => formData.append('files', file))
      const result = await api(
        `/api/applicants/bulk-upload?job_id=${encodeURIComponent(jobId)}`,
        { method: 'POST', body: formData },
      )
      if (requestId !== screeningRequestRef.current) return
      setUploadReport(result)
      setSelectedFiles([])
      await refreshApplicants(jobId)
      await loadData()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const refreshApplicants = async jobId => {
    const requestId = ++screeningRequestRef.current
    try {
      const rows = await api(`/api/applicants/jobs/${encodeURIComponent(jobId)}`)
      if (requestId !== screeningRequestRef.current) return false
      setScreeningApplicants(rows)
      return true
    } catch (err) {
      if (requestId !== screeningRequestRef.current) return false
      setError(err.message)
      return false
    }
  }

  const refreshScreeningHistoryAndShortlist = async jobId => {
    const requestId = ++screeningRequestRef.current
    const [runs, shortlist] = await Promise.all([
      api(`/api/applicants/jobs/${encodeURIComponent(jobId)}/screening-runs`),
      api(`/api/applicants/jobs/${encodeURIComponent(jobId)}/shortlist`),
    ])
    if (requestId !== screeningRequestRef.current) return false
    setScreeningRuns(runs)
    setShortlistedApplicants(shortlist)
    return true
  }

  const rankJobApplicants = async () => {
    if (!screeningJobId) return
    const jobId = screeningJobId
    const requestId = ++screeningRequestRef.current
    setBusy(true)
    setError('')
    try {
      const result = await api(
        `/api/applicants/jobs/${encodeURIComponent(jobId)}/rank`,
        { method: 'POST' },
      )
      if (requestId !== screeningRequestRef.current) return
      setRanking(result)
      await refreshScreeningHistoryAndShortlist(jobId)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const openSavedRun = async runId => {
    const requestId = ++screeningRequestRef.current
    setBusy(true)
    setError('')
    try {
      const run = await api(`/api/applicants/screening-runs/${encodeURIComponent(runId)}`)
      if (requestId !== screeningRequestRef.current) return
      const job = jobs.find(row => row.id === run.job_id)
      const results = (run.results || []).map(row => ({
        ...(row.evidence || {}),
        applicant_id: row.applicant_id,
        rank: row.rank,
        ranking_score: row.ranking_score,
      }))
      setRanking({
        job_id: run.job_id,
        job_title: job?.title || 'Saved screening',
        screening_run_id: run.id,
        ranked_applicant_count: results.length,
        skipped_applicant_count: 0,
        results,
      })
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const addToShortlist = async applicant => {
    if (!screeningJobId) return
    const jobId = screeningJobId
    const requestId = ++screeningRequestRef.current
    setBusy(true)
    setError('')
    try {
      await api(`/api/applicants/jobs/${jobId}/applicants/${applicant.applicant_id}/shortlist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      if (requestId !== screeningRequestRef.current) return
      await refreshScreeningHistoryAndShortlist(jobId)
      noticeFor('Applicant added to this job’s shortlist.')
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const removeFromShortlist = async applicantId => {
    if (!screeningJobId) return
    const jobId = screeningJobId
    const requestId = ++screeningRequestRef.current
    setBusy(true)
    setError('')
    try {
      await api(`/api/applicants/jobs/${jobId}/applicants/${applicantId}/shortlist`, {
        method: 'DELETE',
      })
      if (requestId !== screeningRequestRef.current) return
      await refreshScreeningHistoryAndShortlist(jobId)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const activeJobs = jobs.filter(job => job.status === 'active')
  const applicantCount = applicants.length
  const completedCount = applicants.filter(
    applicant => applicant.processing_status === 'completed',
  ).length

  if (!currentUser) {
    return <AuthScreen
      mode={authMode}
      values={authValues}
      busy={authBusy || loading}
      error={error}
      onChange={(key, value) => setAuthValues(current => ({ ...current, [key]: value }))}
      onSubmit={submitAuth}
      onToggle={() => {
        setError('')
        setAuthMode(mode => mode === 'login' ? 'register' : 'login')
      }}
    />
  }

  const navItems = [
    { label: 'Overview', icon: Activity },
    { label: 'Jobs', icon: BriefcaseBusiness },
    { label: 'Applicant Screening', icon: Users },
  ]

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark"><BrainCircuit size={23}/></div><div><strong>Talent<span>IQ</span></strong><small>RECRUITER WORKSPACE</small></div></div>
        <div className="workspace-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {navItems.map(({ label, icon: Icon }) => <button key={label} className={`nav-link ${page === label ? 'active' : ''}`} onClick={() => { setPage(label); setError('') }}>
            <Icon size={18}/><span>{label}</span>
          </button>)}
        </nav>
        <div className="sidebar-bottom"><div className="status-dot"/><div><strong>Private recruiter area</strong><small>Signed in as recruiter</small></div></div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">Workspace <ChevronRight size={14}/> <strong>{page}</strong></div>
          <div className="top-actions">
            <button className="icon-button" onClick={() => { setLoading(true); loadData() }} title="Refresh data" aria-label="Refresh data"><RefreshCw size={17}/></button>
            <div className="avatar">{(currentUser.full_name || 'R').slice(0, 1).toUpperCase()}</div>
            <div className="profile-label"><strong>{currentUser.full_name}</strong><small>Recruiter</small></div>
            <button className="secondary-button" onClick={logout}><LogOut size={15}/> Log out</button>
          </div>
        </header>

        <div className="page-content">
          {error && <div className="alert error" role="alert"><AlertCircle size={18}/><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error"><X size={16}/></button></div>}
          {notice && <div className="alert success" role="status"><CheckCircle2 size={18}/><span>{notice}</span></div>}

          {page === 'Overview' && <>
            <section className="welcome-row">
              <div><div className="eyebrow"><Activity size={14}/> RECRUITER WORKSPACE</div><h1>Good work starts with <span>great matches.</span></h1><p>Manage your jobs, screen applicant batches, and review job-specific rankings.</p></div>
              <button className="primary-button" onClick={() => { setPage('Jobs'); openNewJob() }}><Plus size={17}/> Create a job</button>
            </section>
            <section className="stats-grid">
              <Stat icon={BriefcaseBusiness} label="Your jobs" value={jobs.length} note="Owned by your account" color="violet"/>
              <Stat icon={Users} label="Applicants" value={applicantCount} note="Across your jobs" color="blue"/>
              <Stat icon={FileText} label="Processed resumes" value={completedCount} note="Ready for ranking" color="green"/>
              <Stat icon={BrainCircuit} label="Active jobs" value={activeJobs.length} note="Open for screening" color="amber"/>
            </section>
            <section className="content-card">
              <div className="card-heading"><div><h2>Your recent jobs</h2><p>Open a job's applicant pool to upload resumes or run a ranking.</p></div><button className="text-button" onClick={() => setPage('Jobs')}>View all <ChevronRight size={15}/></button></div>
              <JobTable jobs={jobsWithCounts.slice(0, 5)} loading={loading} busy={busy} onEdit={openEditJob} onScreen={openScreening} onStatus={changeJobStatus}/>
            </section>
          </>}

          {page === 'Jobs' && <>
            <PageHeading eyebrow="JOB MANAGEMENT" title="Your jobs" description="Create, edit, close, and screen applicants for jobs owned by your account." action={<button className="primary-button" onClick={openNewJob}><Plus size={17}/> Create a job</button>}/>
            <div className="toolbar"><div className="search-box"><Search size={17}/><input aria-label="Search jobs" placeholder="Search title, company, or location..." value={search} onChange={event => setSearch(event.target.value)}/></div><span className="muted">{filteredJobs.length} jobs</span></div>
            <section className="content-card"><JobTable jobs={jobsWithCounts.filter(job => filteredJobs.some(row => row.id === job.id))} loading={loading} busy={busy} onEdit={openEditJob} onScreen={openScreening} onStatus={changeJobStatus}/></section>
          </>}

          {page === 'Applicant Screening' && <>
            <PageHeading eyebrow="JOB-SPECIFIC SCREENING" title="Applicant Screening" description="Upload multiple resumes for a selected job, review the applicant pool, and rank processed applicants."/>
            <section className="content-card upload-card">
              <div className="card-heading"><div><h2>Choose a job and upload resumes</h2><p>PDF or DOCX, up to 5 MB each and 50 files per batch.</p></div><div className="upload-icon"><Upload size={20}/></div></div>
              <label>Active job
                <select value={screeningJobId} onChange={event => openScreening(event.target.value)}>
                  <option value="">Select a job</option>
                  {activeJobs.map(job => <option key={job.id} value={job.id}>{job.title} · {job.company_name} (#{job.id})</option>)}
                </select>
              </label>
              {screeningJobId && <form className="upload-form" onSubmit={uploadBatch}>
                <label className="file-drop"><Upload size={22}/><strong>Select multiple resumes</strong><span>PDF or DOCX · 5 MB each · 50 maximum</span>
                  <input type="file" multiple accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document" onChange={chooseFiles}/>
                </label>
                {selectedFiles.length > 0 && <div className="resume-list" aria-label="Selected resumes">
                  {selectedFiles.map((file, index) => <div className="resume-row" key={`${file.name}-${file.lastModified}-${index}`}>
                    <div className="resume-file"><FileText size={18}/></div><div className="resume-info"><strong>{file.name}</strong><span>{(file.size / (1024 * 1024)).toFixed(2)} MB</span></div>
                    <button type="button" className="secondary-button" aria-label={`Remove ${file.name}`} onClick={() => setSelectedFiles(rows => rows.filter((_, rowIndex) => rowIndex !== index))}>Remove</button>
                  </div>)}
                </div>}
                <button className="primary-button" disabled={busy || selectedFiles.length === 0}>{busy ? 'Uploading and processing...' : 'Upload and process batch'}</button>
              </form>}
              {!activeJobs.length && <p className="helper-text">Create an active job before screening resumes.</p>}
            </section>

            {uploadReport && <section className="content-card">
              <div className="card-heading"><div><h2>Batch processing</h2><p>{uploadReport.completed} completed · {uploadReport.failed} failed of {uploadReport.total_files} files</p></div><button className="primary-button" disabled={busy || uploadReport.completed === 0} onClick={rankJobApplicants}>{busy ? 'Ranking...' : 'Rank applicants'}</button></div>
              <UploadOutcomes rows={uploadReport.results || []}/>
            </section>}

            {screeningJobId && <section className="content-card">
              <div className="card-heading"><div><h2>Applicant pool</h2><p>{screeningApplicants.length} applicants for this job · rank includes all successfully processed resumes</p></div><div className="screening-actions"><button className="secondary-button" onClick={() => refreshApplicants(screeningJobId)}><RefreshCw size={15}/> Refresh</button><button className="primary-button" disabled={busy || !screeningApplicants.some(row => row.processing_status === 'completed')} onClick={rankJobApplicants}>{busy ? 'Ranking...' : 'Rank applicants'}</button></div></div>
              <ApplicantTable rows={screeningApplicants} loading={busy && screeningApplicants.length === 0}/>
            </section>}

            {screeningJobId && <section className="content-card">
              <div className="card-heading"><div><h2>Saved screening runs</h2><p>Reopen a previous ranking for this job.</p></div><button className="secondary-button" onClick={() => refreshScreeningHistoryAndShortlist(screeningJobId)}><RefreshCw size={15}/> Refresh</button></div>
              {!screeningRuns.length
                ? <Empty title="No saved screenings yet" text="Rank this applicant pool to save a screening run."/>
                : <div className="resume-list">{screeningRuns.map(run => <div className="resume-row" key={run.id}>
                  <div className="resume-file"><BrainCircuit size={18}/></div><div className="resume-info"><strong>Screening #{run.id}</strong><span>{new Date(run.created_at).toLocaleString()} · {run.result_count} ranked applicants</span></div>
                  <button className="secondary-button" disabled={busy} onClick={() => openSavedRun(run.id)}>View results</button>
                </div>)}</div>}
            </section>}

            {screeningJobId && <section className="content-card">
              <div className="card-heading"><div><h2>Shortlist</h2><p>{shortlistedApplicants.length} applicants saved for this job.</p></div></div>
              {!shortlistedApplicants.length
                ? <Empty title="No shortlisted applicants" text="Add applicants from a ranking when you are ready to review them further."/>
                : <div className="resume-list">{shortlistedApplicants.map(applicant => <div className="resume-row" key={applicant.id}>
                  <div className="resume-file"><Users size={18}/></div><div className="resume-info"><strong>{applicant.filename}</strong><span>{[applicant.role || 'Role not found', applicant.years_experience == null ? 'Experience not found' : `${applicant.years_experience} years`, applicant.seniority || 'Seniority not found', applicant.education || 'Education not found'].join(' · ')}{applicant.note ? ` · ${applicant.note}` : ''}</span></div>
                  <button className="secondary-button" disabled={busy} onClick={() => removeFromShortlist(applicant.applicant_id)}>Remove</button>
                </div>)}</div>}
            </section>}

            {ranking && <section className="content-card ranking-card">
              <div className="card-heading"><div><h2>{ranking.job_title || 'Applicant ranking'}</h2><p>{ranking.ranked_applicant_count} ranked · {ranking.skipped_applicant_count} skipped</p></div><span className="count-chip">Job #{ranking.job_id}</span></div>
              <div className="score-notice"><AlertCircle size={17}/><span>Scores are raw model scores, not probabilities. Review each resume and the qualification evidence before making a decision.</span></div>
              {ranking.screening_run_id && <p className="muted">Saved screening run #{ranking.screening_run_id}</p>}
              <RankedApplicants rows={ranking.results || []} shortlistedIds={new Set(shortlistedApplicants.map(row => row.applicant_id))} busy={busy} onShortlist={addToShortlist}/>
              {ranking.skipped_applicants?.length > 0 && <details><summary>{ranking.skipped_applicants.length} skipped applicants</summary><ul>{ranking.skipped_applicants.map(row => <li key={row.applicant_id}>{row.filename}: {row.reason}</li>)}</ul></details>}
            </section>}
          </>}
        </div>
      </main>

      {showJobForm && <JobModal
        job={editingJob}
        values={jobForm}
        busy={busy}
        onChange={(key, value) => setJobForm(current => ({ ...current, [key]: value }))}
        onClose={() => setShowJobForm(false)}
        onSubmit={saveJob}
      />}
    </div>
  )
}

function AuthScreen({ mode, values, busy, error, onChange, onSubmit, onToggle }) {
  const registering = mode === 'register'
  return <main className="auth-shell">
    <section className="auth-panel">
      <div className="brand auth-brand"><div className="brand-mark"><BrainCircuit size={23}/></div><div><strong>Talent<span>IQ</span></strong><small>RECRUITER WORKSPACE</small></div></div>
      <div className="eyebrow">PRIVATE RECRUITER PLATFORM</div>
      <h1>{registering ? 'Create your recruiter account' : 'Welcome back'}</h1>
      <p>{registering ? 'Register to manage jobs and screen applicants.' : 'Sign in to access your jobs and applicant screening.'}</p>
      {error && <div className="alert error" role="alert"><AlertCircle size={18}/><span>{error}</span></div>}
      <form className="auth-form" onSubmit={onSubmit}>
        {registering && <label>Full name<input autoComplete="name" required maxLength={255} value={values.full_name} onChange={event => onChange('full_name', event.target.value)}/></label>}
        <label>Email<input type="email" autoComplete="email" required value={values.email} onChange={event => onChange('email', event.target.value)}/></label>
        <label>Password<input type="password" autoComplete={registering ? 'new-password' : 'current-password'} minLength={registering ? 12 : 1} maxLength={1024} required value={values.password} onChange={event => onChange('password', event.target.value)}/>{registering && <small>Use at least 12 characters.</small>}</label>
        <button className="primary-button" disabled={busy}>{busy ? 'Please wait...' : registering ? 'Create recruiter account' : 'Sign in'}</button>
      </form>
      <p className="auth-switch">{registering ? 'Already registered?' : 'New recruiter?'} <button type="button" className="text-button" onClick={onToggle}>{registering ? 'Sign in' : 'Create an account'}</button></p>
    </section>
  </main>
}

function JobTable({ jobs, loading, busy, onEdit, onScreen, onStatus }) {
  if (loading) return <div className="loading-state"><RefreshCw size={19} className="spin"/> Loading your jobs...</div>
  if (!jobs.length) return <Empty title="No jobs found" text="Create a job to start screening applicants."/>
  return <div className="table-wrap"><table><thead><tr><th>JOB</th><th>LOCATION</th><th>APPLICANTS</th><th>STATUS</th><th>ACTIONS</th></tr></thead><tbody>
    {jobs.map(job => <tr key={job.id}>
      <td><div className="job-title-cell"><div className="job-symbol"><BriefcaseBusiness size={17}/></div><div><strong>{job.title}</strong><span>{job.company_name} · #{job.id}</span></div></div></td>
      <td>{job.location || '—'}</td>
      <td>{job.applicant_count ?? '—'}</td>
      <td><span className={`status-pill ${job.status === 'active' ? 'active-status' : 'neutral'}`}>{job.status}</span></td>
      <td><div className="table-actions"><button className="table-action" onClick={() => onEdit(job)}>Edit</button><button className="table-action" onClick={() => onScreen(job.id)} disabled={job.status !== 'active'}>Screen</button><button className="table-action" onClick={() => onStatus(job)} disabled={busy}>{job.status === 'active' ? 'Close' : 'Reopen'}</button></div></td>
    </tr>)}
  </tbody></table></div>
}

function ApplicantTable({ rows, loading }) {
  if (loading) return <div className="loading-state"><RefreshCw size={18} className="spin"/> Loading applicant pool...</div>
  if (!rows.length) return <Empty title="No applicants yet" text="Upload resumes above to create the job's applicant pool."/>
  return <div className="resume-list">{rows.map(row => <div className="resume-row" key={row.id}>
    <div className="resume-file"><FileText size={18}/></div><div className="resume-info"><strong>{row.filename}</strong><span>{[row.role || 'Role not found', row.years_experience == null ? 'Experience not found' : `${row.years_experience} years`, row.seniority || 'Seniority not found', row.education || 'Education not found'].join(' · ')}</span></div>
    <span className={`status-pill ${row.processing_status === 'completed' ? 'active-status' : 'neutral'}`}>{row.processing_status}</span>
  </div>)}</div>
}

function UploadOutcomes({ rows }) {
  if (!rows.length) return <Empty title="No upload outcomes" text="The server returned an empty batch."/>
  return <div className="resume-list">{rows.map((row, index) => <div className="resume-row" key={`${row.filename}-${index}`}>
    <div className="resume-file"><FileText size={18}/></div><div className="resume-info"><strong>{row.filename}</strong><span>{row.status === 'completed' ? [row.role, row.years_experience == null ? null : `${row.years_experience} years`, row.seniority, row.education].filter(Boolean).join(' · ') || 'Some fields were not found in this resume.' : row.message}</span></div>
    <span className={`status-pill ${row.status === 'completed' ? 'active-status' : 'neutral'}`}>{row.status}</span>
  </div>)}</div>
}

function RankedApplicants({ rows, shortlistedIds, busy, onShortlist }) {
  if (!rows.length) return <Empty title="No applicants could be ranked" text="Confirm applicants have completed processing and an AI profile."/>
  return rows.map(row => {
    const matchedRequired = row.matched_skills || []
    const missingRequired = row.missing_skills || []
    const matchedPreferred = row.matched_preferred_skills || []
    const missingPreferred = row.missing_preferred_skills || []
    const coverage = row.skill_coverage == null || !Number.isFinite(Number(row.skill_coverage))
      ? null
      : Math.max(0, Math.min(1, Number(row.skill_coverage)))
    const signals = [
      ['Role compatibility', row.role_compatibility],
      ['Experience compatibility', row.experience_compatibility],
      ['Seniority compatibility', row.seniority_compatibility],
      ['Education compatibility', row.education_compatibility],
    ]

    return <article className="ranked-candidate" key={row.applicant_id}>
      <div className="rank-number">#{row.rank}</div>
      <div className="ranked-info">
        <strong>{row.filename || `Applicant ${row.applicant_id}`}</strong>
        <span>{[row.role || 'Role not found', row.years_experience == null ? 'Experience not found' : `${row.years_experience} years`, row.seniority || 'Seniority not found', row.education || 'Education not found'].join(' ? ')}</span>

        <div className="coverage-block">
          <div className="coverage-heading"><span>Required skill coverage</span><strong>{coverage == null ? 'Unavailable' : `${Math.round(coverage * 100)}%`}</strong></div>
          {coverage != null && <div className="coverage-track" role="img" aria-label={`Required skill coverage ${Math.round(coverage * 100)} percent`}><span style={{ width: `${coverage * 100}%` }}/></div>}
        </div>

        <SkillEvidence title="Required skills" matched={matchedRequired} missing={missingRequired}/>
        <SkillEvidence title="Preferred skills" matched={matchedPreferred} missing={missingPreferred}/>

        <div className="match-evidence-heading">Compatibility inputs (0?1; not individual model contributions)</div>
        <div className="compatibility-grid" aria-label="Qualification compatibility inputs">
          {signals.map(([label, value]) => <div className="compatibility-item" key={label}>
            <span>{label}</span>
            <strong>{value == null || !Number.isFinite(Number(value)) ? 'Unavailable' : Number(value).toFixed(2)}</strong>
          </div>)}
        </div>
      </div>
      <div className="ranked-score">
        <strong>{Number.isFinite(Number(row.ranking_score)) ? Number(row.ranking_score).toFixed(3) : '?'}</strong>
        <span>Raw model score</span>
        <button className="secondary-button" disabled={busy || shortlistedIds.has(row.applicant_id)} onClick={() => onShortlist(row)}>{shortlistedIds.has(row.applicant_id) ? 'Shortlisted' : 'Add to shortlist'}</button>
      </div>
    </article>
  })
}

function SkillEvidence({ title, matched, missing }) {
  return <section className="skill-section" aria-label={`${title} comparison`}>
    <div className="skill-section-title">{title} <span>{matched.length} matched ? {missing.length} missing</span></div>
    {(matched.length > 0 || missing.length > 0) ? <div className="skill-tags">
      {matched.map(skill => <span className="skill-tag matched" key={`matched-${skill}`}>Matched: {skill}</span>)}
      {missing.map(skill => <span className="skill-tag missing" key={`missing-${skill}`}>Missing: {skill}</span>)}
    </div> : <p className="muted">No {title.toLowerCase()} listed for comparison.</p>}
  </section>
}

function JobModal({ job, values, busy, onChange, onClose, onSubmit }) {
  return <div className="modal-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}>
    <section className="modal" role="dialog" aria-modal="true" aria-labelledby="job-modal-title">
      <div className="modal-heading"><div><div className="eyebrow">JOB REQUIREMENTS</div><h2 id="job-modal-title">{job ? 'Edit job' : 'Create a job'}</h2><p>Structured requirements are sent to the matching pipeline.</p></div><button className="icon-button" onClick={onClose} aria-label="Close dialog"><X size={19}/></button></div>
      <form className="job-form" onSubmit={onSubmit}>
        <div className="form-two"><label>Job title<input required maxLength={255} value={values.title} onChange={event => onChange('title', event.target.value)}/></label><label>Company<input required maxLength={255} value={values.company_name} onChange={event => onChange('company_name', event.target.value)}/></label></div>
        <div className="form-two"><label>Location<input value={values.location} onChange={event => onChange('location', event.target.value)}/></label><label>Industry<input value={values.industry} onChange={event => onChange('industry', event.target.value)}/></label></div>
        <label>Job description<textarea required rows="5" value={values.description} onChange={event => onChange('description', event.target.value)}/></label>
        <label>Required skills<input value={values.required_skills} onChange={event => onChange('required_skills', event.target.value)} placeholder="Python, SQL, Machine Learning"/></label>
        <label>Preferred skills<input value={values.preferred_skills} onChange={event => onChange('preferred_skills', event.target.value)} placeholder="Docker, AWS"/></label>
        <div className="form-two"><label>Minimum experience (years)<input type="number" min="0" step="1" value={values.experience_required} onChange={event => onChange('experience_required', event.target.value)}/></label><label>Seniority<input value={values.seniority_required} onChange={event => onChange('seniority_required', event.target.value)} placeholder="Entry / Mid / Senior"/></label></div>
        <label>Education requirement<input value={values.education_required} onChange={event => onChange('education_required', event.target.value)}/></label>
        <div className="modal-actions"><button type="button" className="secondary-button" onClick={onClose}>Cancel</button><button className="primary-button" disabled={busy}>{busy ? 'Saving...' : job ? 'Save changes' : 'Create and process job'}</button></div>
      </form>
    </section>
  </div>
}

function PageHeading({ eyebrow, title, description, action }) {
  return <section className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action}</section>
}

function Stat({ icon: Icon, label, value, note, color }) {
  return <div className="stat-card"><div className={`stat-icon ${color}`}><Icon size={20}/></div><div className="stat-label">{label}</div><div className="stat-value">{value}</div><div className="stat-note">{note}</div></div>
}

function Empty({ title, text }) {
  return <div className="empty-state"><div className="empty-icon"><FileText size={20}/></div><strong>{title}</strong><p>{text}</p></div>
}

export default RecruiterPortal
