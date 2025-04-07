
var video_names = ['child-field','cat-spinning','dog-snow','hiking-follow','dog-running','blackswan','dogs-jump','drift-straight', 'dance-twirl', 'bmx-trees', 'motocross-jump', 'soapbox', 'kite-surf', 'horsejump-high'];
var download_paths = [
  'data/high_res/child-field.mp4',
  'data/high_res/cat-spinning.mp4',
  'data/high_res/dog-snow.mp4',
  'data/high_res/hiking-follow.mp4',
  'data/high_res/dog-running.mp4',
  'data/high_res/blackswan.mp4',
  'data/high_res/dogs-jump.mp4',
  'data/high_res/drift-straight.mp4', 
  'data/high_res/dance-twirl.mp4', 
  'data/high_res/bmx-trees.mp4',
  'data/high_res/motocross-jump.mp4',
  'data/high_res/soapbox.mp4',
  'data/high_res/kite-surf.mp4',
  'data/high_res/horsejump-high.mp4'
];
var videos = [];

var video_width = 864;
const VIDEO_ASPECT_RATIO = 16.0 / 9.0;

$(function() {
    var canvas = document.getElementById('canvas');
    var ctx = canvas.getContext('2d');

    current_video_idx = 0;

    thumbnails = [
      document.getElementById('thumb-0'),
      document.getElementById('thumb-1'),
      document.getElementById('thumb-2'),
      document.getElementById('thumb-3'),
      document.getElementById('thumb-4'),
      document.getElementById('thumb-5'),
      document.getElementById('thumb-6'),
      document.getElementById('thumb-7'),
      document.getElementById('thumb-8'),
      document.getElementById('thumb-9'),
      document.getElementById('thumb-10'),
      document.getElementById('thumb-11'),
      document.getElementById('thumb-12'),
      document.getElementById('thumb-13'),    
    ];
    for (var i = 0; i < thumbnails.length; i++) {
      thumbnails[i].addEventListener('click', change_video_index.bind(this, i));
    }
    
    var canvas_overlay = document.getElementById('canvas-overlay');
    var ctx_overlay = canvas_overlay.getContext('2d');

    if (videos.length == 0) {
      load_videos();
    };

    (function loop() {
      video = videos[current_video_idx]
      ctx.drawImage(video, 0, 0, 864, 480, 0, 0, video_width, video_width/VIDEO_ASPECT_RATIO);
      ctx_overlay.drawImage(video, 864, 0, 864, 480, 0, 0, video_width, video_width/VIDEO_ASPECT_RATIO);
      setTimeout(loop, 1000 / 60); // drawing at 30fps
      set_play_pause_icon();
    })();

  });
  
function change_video_index (idx) {
  thumbnails[idx].classList.add("active-btn");
  if (current_video_idx != idx) {
    thumbnails[current_video_idx].classList.remove("active-btn");
  }
  videos[current_video_idx].pause()
  current_video_idx = idx;
  current_video = videos[current_video_idx]
  current_video.currentTime = 0;
  current_video.play();
  set_play_pause_icon();
}

function fullscreen() {
  current_video = videos[current_video_idx]
  current_video.style.visibility = "visible";
  const fullscreenElement =
    document.fullscreenElement ||
    document.mozFullScreenElement ||
    document.webkitFullscreenElement ||
    document.msFullscreenElement;
  if (fullscreenElement) {
    exitFullscreen();
  } else {
    launchIntoFullscreen(current_video);
  }
}

function download() {
  current_video = videos[current_video_idx]
  var link = document.createElement('a');
  link.download = video_names[current_video_idx] + '.mp4';
  link.href = download_paths[current_video_idx];
  link.click();
}

function launchIntoFullscreen(element) {
  if (element.requestFullscreen) {
    element.requestFullscreen();
  } else if (element.mozRequestFullScreen) {
    element.mozRequestFullScreen();
  } else if (element.webkitRequestFullscreen) {
    element.webkitRequestFullscreen();
  } else if (element.msRequestFullscreen) {
    element.msRequestFullscreen();
  } else {
    element.classList.toggle('fullscreen');
  }
}

function exitFullscreen() {
  if (document.exitFullscreen) {
    document.exitFullscreen();
  } else if (document.mozCancelFullScreen) {
    document.mozCancelFullScreen();
  } else if (document.webkitExitFullscreen) {
    document.webkitExitFullscreen();
  }
}

if (document.addEventListener)
{
 document.addEventListener('fullscreenchange', exitHandler, false);
 document.addEventListener('mozfullscreenchange', exitHandler, false);
 document.addEventListener('MSFullscreenChange', exitHandler, false);
 document.addEventListener('webkitfullscreenchange', exitHandler, false);
}

function exitHandler()
{
 if (!document.webkitIsFullScreen && !document.mozFullScreen && !document.msFullscreenElement)
 {
  current_video = videos[current_video_idx]
  current_video.style.visibility = "hidden";
 }
}

function load_videos() {
  for (var i = 0; i < video_names.length; i++) {
    videos.push(document.getElementById(video_names[i])); 
  }
}

function set_play_pause_icon() {
  button = document.getElementById('play-btn')
  current_video = videos[current_video_idx]
  if (current_video.paused) {
    button.classList.remove("fa-pause");
    button.classList.add("fa-play");
  } else {
    button.classList.add("fa-pause");
    button.classList.remove("fa-play");
  }
}

function play_pause() {
  current_video = videos[current_video_idx]
  if (current_video.paused) {
    current_video.play();
  } else {
    current_video.pause();
  }
  set_play_pause_icon();
}

function resize_canvas() {
  var canvas = document.getElementById('canvas');
  var canvas_overlay = document.getElementById('canvas-overlay');
  var main_results = document.getElementById('main-results');

  var width = main_results.offsetWidth;
  var height = width / VIDEO_ASPECT_RATIO;

  main_results.style.height = height;

  video_width = width;
  canvas.width = width;
  canvas.height = height;
  canvas.style.width = width;
  canvas.style.height = height;
  canvas_overlay.width = width;
  canvas_overlay.height = height;
  canvas_overlay.style.width = width;
  canvas_overlay.style.height = height;
}

window.onload = function() {
  const root = document.documentElement;
  const checkbox = document.getElementById('opacity-toggle')

  load_videos();
  checkbox.addEventListener('change', (event) => {
    if (event.currentTarget.checked) {
      root.style.setProperty("--opacity", `100%`);
    } else {
      root.style.setProperty("--opacity", `0%`);
    }
  })

  change_video_index(0);
  videos[0].play();

  const hoverImage = document.getElementById('hover-image');
  const gptQueries = document.querySelectorAll('.gpt-query');

  gptQueries.forEach(query => {
    query.addEventListener('mouseover', () => {
      hoverImage.src = 'data/gpt_example/' + query.id + '.jpg';
    });
    query.addEventListener('mouseout', () => {
      hoverImage.src = 'data/gpt_example/base.jpg';
    });
  });

}

window.addEventListener('resize', resize_canvas, false);

document.addEventListener("DOMContentLoaded", function() {
  resize_canvas();
});


function slide_left() {
  slider_window = document.getElementById('thumbnails-scroll');
  slider_window.scrollLeft = 0;
}

function slide_right() {
  slider_window = document.getElementById('thumbnails-scroll');
  slider_window.scrollLeft += 1000;
}
