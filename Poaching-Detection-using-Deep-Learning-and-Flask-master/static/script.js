//const videoFileInput = document.getElementById('video-file');
const video = document.getElementById("video");
const detectBtn = document.getElementById("detect-btn");
const status1 = document.getElementById("status1");

//const fileInput = document.getElementById('video-file');
fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  const filePath = URL.createObjectURL(file);
  console.log("Selected file:", file.name);
  console.log("File path:", filePath);
  // You can store the file path in a variable for further use
  // For example:
  // const videoPlayer = document.getElementById('videoPlayer');
  // videoPlayer.src = filePath;
});

function detectPoaching() {
  if (!video.src) {
    status1.innerHTML = "Please upload a video file.";
    return;
  }
  console.log(filePath);
  // Implement your poaching detection algorithm here
  // and display the result in the status element

  status1.innerHTML = `$filePath`;
}

videoFileInput.addEventListener("change", function () {
  if (this.files && this.files[0]) {
    video.src = URL.createObjectURL(this.files[0]);
    //video.play();

    document.querySelector(".video-container").style.display = "block";
  }
});

//new 10:50pm
function handleFile() {
  const fileInput = document.getElementById("myFileInput");
  const file = fileInput.files[0];

  console.log(file.name); // log the name of the selected file

  // perform further actions with the file object here
}
