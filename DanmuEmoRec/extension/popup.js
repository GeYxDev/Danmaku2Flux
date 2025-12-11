document.addEventListener('DOMContentLoaded', function() {
  const btn = document.getElementById('recommend-btn');
  const list = document.getElementById('result-list');
  const loading = document.getElementById('loading');
  const info = document.getElementById('current-info');

  // Get the current tab URL
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    const currentUrl = tabs[0].url;
    
    // Extract BV number
    const bvRegex = /(BV[a-zA-Z0-9]{10})/;
    const match = currentUrl.match(bvRegex);

    if (match) {
      const bvid = match[1];
      info.textContent = `Current video: ${bvid}`;
      
      // Bind click event
      btn.addEventListener('click', () => {
        fetchRecommendation(bvid);
      });
    } else {
      info.textContent = "No video link detected";
      btn.disabled = true;
    }
  });

  // Send requests like the backend
  async function fetchRecommendation(bvid) {
    btn.disabled = true;
    list.innerHTML = '';
    loading.style.display = 'block';

    try {
      const response = await fetch(`http://127.0.0.1:8000/recommend?bvid=${bvid}`);
      const result = await response.json();

      loading.style.display = 'none';
      btn.disabled = false;

      if (result.status === 'success') {
        renderList(result.data);
      } else {
        list.innerHTML = `<li style="color:red">An error occurred: ${result.detail}</li>`;
      }
    } catch (error) {
      console.error(error);
      loading.style.display = 'none';
      btn.disabled = false;
      list.innerHTML = `<li style="color:red">Unable to communicate with the recommendation system (╥﹏╥)</li>`;
    }
  }

  // Rendering recommendation list
  function renderList(videos) {
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