document.addEventListener('DOMContentLoaded', function() {
  const btn = document.getElementById('recommend-btn');
  const list = document.getElementById('result-list');
  const loading = document.getElementById('loading');
  const info = document.getElementById('current-info');

  // Maximum number of historical cache records
  const MAX_CACHE_SIZE = 20;

  // Get the current tab URL
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    const currentUrl = tabs[0].url;
    
    // Extract BV number
    const bvRegex = /(BV[a-zA-Z0-9]{10})/;
    const match = currentUrl.match(bvRegex);

    if (match) {
      const bvid = match[1];
      info.textContent = `Current video: ${bvid}`;

      // Match records from cache first when opening a webpage
      checkHistoryCache(bvid);
      
      // Bind click event
      btn.addEventListener('click', () => {
        fetchRecommendation(bvid);
      });
    } else {
      info.textContent = "No video link detected";
      btn.disabled = true;
    }
  });

  // Check historical recommendation records
  function checkHistoryCache(bvid) {
    // Read the array that stores historical records
    chrome.storage.local.get({ videoHistory: [] }, function(result) {
      const history = result.videoHistory;

      // Check if the current BV number exists in the array
      const cachedItem = history.find(item => item.bvid === bvid);

      // If the historical record exists, return
      if (cachedItem) {
        renderList(cachedItem.data);
      }
    });
  }

  // Save recommendation records locally
  function saveToHistory(bvid, data) {
    // Read the array that stores historical records
    chrome.storage.local.get({ videoHistory: [] }, function(result) {
      let history = result.videoHistory;

      // Cache query records for deduplication
      history = history.filter(item => item.bvid !== bvid);

      // If the record already exists, update it to the most recent visit
      history.unshift({
        bvid: bvid,
        data: data,
        timestamp: new Date().getTime()
      });

      // Abandoning old data
      if (history.length > MAX_CACHE_SIZE) {
        history = history.slice(0, MAX_CACHE_SIZE);
      }

      // Write historical recommendation data locally
      chrome.storage.local.set({ videoHistory: history }, function() {
        console.log(`[Cache] Saved. Total history count: ${history.length}`);
      });
    });
  }

  // Send requests like the backend
  async function fetchRecommendation(bvid) {
    btn.disabled = true;
    list.innerHTML = '';
    loading.style.display = 'block';
    loading.textContent = "Extracting emotions & Searching...";

    try {
      const response = await fetch(`http://127.0.0.1:8000/recommend?bv=${bvid}`);
      const result = await response.json();

      loading.style.display = 'none';
      btn.disabled = false;

      if (response.ok) {
        if (result.code === 200 && result.data.length > 0) {
           renderList(result.data);

           // Save the obtained recommendation data
           saveToHistory(bvid, result.data);
        } else {
           showMessage(result.message || "No similar videos found.", "orange");
        }
      } else {
        const errorMsg = result.message || `Error: ${response.statusText}`;
        if (result.code === 404) {
             showMessage(errorMsg, "#666");
        } else if (result.code === 422) {
             showMessage(errorMsg, "#d9534f");
        } else {
             showMessage("Server Error: " + errorMsg, "red");
        }
      }
    } catch (error) {
      console.error(error);
      loading.style.display = 'none';
      btn.disabled = false;
      list.innerHTML = `<li style="color:red">Unable to communicate with the recommendation system (╥﹏╥)</li>`;
    }
  }

  // Display text information
  function showMessage(msg, color) {
      list.innerHTML = `<li style="color:${color}; padding: 10px; text-align: center;">${msg}</li>`;
  }
  
  // Rendering recommendation list
  function renderList(videos) {
    list.innerHTML = '';
    if (videos.length === 0) {
      list.innerHTML = '<li>No similar videos found...</li>';
      return;
    }

    videos.forEach(video => {
      const li = document.createElement('li');
      li.className = 'video-item';
      // Similarity percentage
      const scorePercent = (video.score * 100).toFixed(1);
      
      li.innerHTML = `
        <a href="${video.link}" target="_blank" title="${video.title}">
          ${video.title}
        </a>
        <div class="score">Emotional similarity: ${scorePercent}%</div>
      `;
      list.appendChild(li);
    });
  }
});