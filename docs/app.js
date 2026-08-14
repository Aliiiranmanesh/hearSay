// ==========================================================================
// HearSayBench Web Application Interactive Logic
// ==========================================================================

let leaderboardData = [];
let scenariosData = [];
let filteredScenarios = [];
let displayedScenariosCount = 12;

let currentSortColumn = 'weighted_score';
let currentSortOrder = 'desc';
let currentLeaderboardFilter = 'all';

document.addEventListener('DOMContentLoaded', () => {
  fetchLeaderboard();
  fetchScenarios();
  setupEventListeners();
});

// --------------------------------------------------------------------------
// 1. Fetch & Render Leaderboard
// --------------------------------------------------------------------------
async function fetchLeaderboard() {
  try {
    const res = await fetch('data/leaderboard.json');
    leaderboardData = await res.json();
    renderLeaderboard();
  } catch (err) {
    console.error('Failed to load leaderboard data:', err);
  }
}

function renderLeaderboard() {
  const tbody = document.getElementById('leaderboard-body');
  const searchVal = document.getElementById('leaderboard-search').value.toLowerCase().trim();

  let filtered = leaderboardData.filter(item => {
    // Filter by provider group
    if (currentLeaderboardFilter === 'Open Source') {
      if (item.badge !== 'Open Weight') return false;
    } else if (currentLeaderboardFilter !== 'all' && item.provider !== currentLeaderboardFilter) {
      return false;
    }

    // Filter by search text
    if (searchVal && !item.model.toLowerCase().includes(searchVal)) {
      return false;
    }

    return true;
  });

  // Sort data
  filtered.sort((a, b) => {
    let valA = a[currentSortColumn];
    let valB = b[currentSortColumn];

    if (typeof valA === 'string') {
      return currentSortOrder === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
    }
    return currentSortOrder === 'asc' ? valA - valB : valB - valA;
  });

  tbody.innerHTML = '';

  filtered.forEach(item => {
    const tr = document.createElement('tr');
    
    // Rank formatting
    let rankDisplay = `#${item.rank}`;


    tr.innerHTML = `
      <td style="font-weight: 700;">${rankDisplay}</td>
      <td>
        <div class="model-cell">
          <span>${item.model}</span>
          ${item.badge ? `<span class="badge-tag">${item.badge}</span>` : ''}
        </div>
      </td>
      <td class="score-cell">${item.situational_comprehension.toFixed(3)}</td>
      <td class="score-cell">${item.capability_freedom.toFixed(3)}</td>
      <td class="score-cell">${item.register_appropriateness.toFixed(3)}</td>
      <td class="score-cell">${item.honesty_uncertainty.toFixed(3)}</td>
      <td class="score-cell highlight-cell">${item.weighted_score.toFixed(3)} <span style="font-size:0.75rem; color: var(--text-dim);">± ${(item.weighted_score - item.ci_lower).toFixed(3)}</span></td>
      <td class="score-cell safety-cell">${item.safety_harm_avg.toFixed(3)}</td>
    `;
    tbody.appendChild(tr);
  });
}

// --------------------------------------------------------------------------
// 2. Fetch & Render Scenarios Explorer
// --------------------------------------------------------------------------
async function fetchScenarios() {
  try {
    const res = await fetch('data/scenarios.json');
    scenariosData = await res.json();
    applyScenarioFilters();
  } catch (err) {
    console.error('Failed to load scenarios data:', err);
  }
}

function applyScenarioFilters() {
  const searchVal = document.getElementById('scenario-search').value.toLowerCase().trim();
  const activeCatBtn = document.querySelector('.cat-btn.active');
  const catFilter = activeCatBtn ? activeCatBtn.dataset.cat : 'all';

  filteredScenarios = scenariosData.filter(item => {
    // Filter by Category
    if (catFilter !== 'all' && item.category !== catFilter) {
      return false;
    }

    // Filter by Search Query
    if (searchVal) {
      const matchScenario = item.scenario.toLowerCase().includes(searchVal);
      const matchPrompt = item.prompt.toLowerCase().includes(searchVal);
      const matchWeird = item.weird_prior.toLowerCase().includes(searchVal);
      const matchImpediment = item.impediment.toLowerCase().includes(searchVal);
      const matchId = item.id.toLowerCase().includes(searchVal);
      const matchSubtype = item.subtype.toLowerCase().includes(searchVal);

      if (!matchScenario && !matchPrompt && !matchWeird && !matchImpediment && !matchId && !matchSubtype) {
        return false;
      }
    }

    return true;
  });

  displayedScenariosCount = 12;
  renderScenarios();
}

function renderScenarios() {
  const grid = document.getElementById('scenarios-grid');
  const counter = document.getElementById('scenarios-counter');
  const loadMoreBtn = document.getElementById('load-more-btn');

  counter.textContent = `Showing ${Math.min(displayedScenariosCount, filteredScenarios.length)} of ${filteredScenarios.length} scenarios`;

  const toRender = filteredScenarios.slice(0, displayedScenariosCount);

  if (toRender.length === 0) {
    grid.innerHTML = `<div class="column is-12 has-text-centered text-muted py-6">No scenarios match your search query.</div>`;
    loadMoreBtn.style.display = 'none';
    return;
  }

  grid.innerHTML = toRender.map(item => `
    <div class="column is-6">
      <div class="scenario-card-box">
        <div class="scenario-header">
          <span class="scenario-id-tag">${item.id}</span>
          <span class="scenario-cat-tag">${item.category} • ${item.subtype}</span>
        </div>
        
        <div>
          <span class="scenario-label">Ground Truth Scenario</span>
          <div style="font-size: 0.92rem; color: #334155; margin-top: 0.2rem;">${item.scenario}</div>
        </div>

        <div>
          <span class="scenario-label">User Query Prompt</span>
          <div class="prompt-box">"${item.prompt}"</div>
        </div>

        <div>
          <span class="scenario-label" style="color: #dc2626;">WEIRD Institutional Prior</span>
          <div class="weird-prior-text">${item.weird_prior}</div>
        </div>

        <div>
          <span class="scenario-label" style="color: #d97706;">Conversion Impediment</span>
          <div class="impediment-text">${item.impediment}</div>
        </div>
      </div>
    </div>
  `).join('');

  if (displayedScenariosCount >= filteredScenarios.length) {
    loadMoreBtn.style.display = 'none';
  } else {
    loadMoreBtn.style.display = 'inline-flex';
  }
}

// --------------------------------------------------------------------------
// 3. Event Listeners Setup
// --------------------------------------------------------------------------
function setupEventListeners() {
  // Leaderboard Provider Filter Buttons
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active', 'is-primary'));
      e.target.classList.add('active', 'is-primary');
      currentLeaderboardFilter = e.target.dataset.filter;
      renderLeaderboard();
    });
  });

  // Leaderboard Search
  document.getElementById('leaderboard-search').addEventListener('input', () => {
    renderLeaderboard();
  });

  // Table Column Sort
  document.querySelectorAll('#leaderboard-table th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const sortCol = th.dataset.sort;
      if (currentSortColumn === sortCol) {
        currentSortOrder = currentSortOrder === 'asc' ? 'desc' : 'asc';
      } else {
        currentSortColumn = sortCol;
        currentSortOrder = 'desc';
      }

      // Update header icons
      document.querySelectorAll('#leaderboard-table th.sortable').forEach(header => {
        header.classList.remove('sorted-asc', 'sorted-desc');
        header.querySelector('i').className = 'fas fa-sort';
      });

      th.classList.add(currentSortOrder === 'asc' ? 'sorted-asc' : 'sorted-desc');
      th.querySelector('i').className = currentSortOrder === 'asc' ? 'fas fa-sort-up' : 'fas fa-sort-down';

      renderLeaderboard();
    });
  });

  // Category Filter Buttons
  document.querySelectorAll('#category-filters .cat-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('#category-filters .cat-btn').forEach(b => b.classList.remove('active', 'is-info'));
      e.target.classList.add('active', 'is-info');
      applyScenarioFilters();
    });
  });

  // Scenario Search Input
  document.getElementById('scenario-search').addEventListener('input', () => {
    applyScenarioFilters();
  });

  // Load More Button
  document.getElementById('load-more-btn').addEventListener('click', () => {
    displayedScenariosCount += 12;
    renderScenarios();
  });
}

// --------------------------------------------------------------------------
// 4. Utility Copy Code
// --------------------------------------------------------------------------
function copyBibtex() {
  const codeText = document.getElementById('bibtex-code').innerText;
  navigator.clipboard.writeText(codeText).then(() => {
    const btn = document.querySelector('.copy-bib-btn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
    setTimeout(() => {
      btn.innerHTML = originalText;
    }, 2000);
  });
}

