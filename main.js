const webcamElement =document.getElementById("webcam");
const canvasElement = document.getElementById("canvas");
const webcam =new webcam(webcamElement , "user", canvasElement);
webcam.start();



// const myModal = new bootstrap.Modal(document.getElementById('myModal'), options)


// $(document).ready(function(){
//     let video = document.getElementById('video');
//     let canvas = document.getElementById('canvas');
//     let context = canvas.getContext('2d');
//     let track = null;
    
//     // Start video stream
//     navigator.mediaDevices.getUserMedia({ video: true })
//     .then(function(stream) {
//         video.srcObject = stream;
//         track = stream.getTracks()[0];
//     })
//     .catch(function(error) {
//         console.log("Error accessing the camera: " + error);
//     });

//     // Continuously capture and send frames to server
//     setInterval(function(){
//         context.drawImage(video, 0, 0, 640, 480);
//         let data = canvas.toDataURL('image/jpeg', 0.5);
//         $.ajax({
//             type: 'POST',
//             url: "http://localhost:5500",
//             data: {
//                 imageBase64: data
//             },
//             success: function(response) {
//                 $('#detected-image').attr('src', 'data:image/jpeg;base64,' + response);
//             }
//         });
//     }, 1000); // Adjust the interval as needed
// });



// onclick function
function onClick() {
    window.location.href = 'login.html';
}

let button = document.getElementById('myButton');
button.addEventListener('click', onClick);

function onClick() {
    window.location.href = 'signin.html';
}

let singninbutton = document.getElementById('Buttonn');
singninbutton.addEventListener('click', onClick);