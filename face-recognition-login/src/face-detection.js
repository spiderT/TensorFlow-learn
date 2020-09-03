require("./assets/styles/index.less");
const faceapi = require("./assets/face-api/face-api.js");
const bbt = require("./assets/face-api/js/bbt");
const commons = require("./assets/face-api/js/commons");
const errorMap = {
  NotAllowedError: "摄像头已被禁用，请在当前浏览器设置中开启后重试",
  AbortError: "硬件问题，导致无法访问摄像头",
  NotFoundError: "未检测到可用摄像头",
  NotReadableError: "操作系统上某个硬件、浏览器或者网页层面发生错误，导致无法访问摄像头",
  OverConstrainedError: "未检测到可用摄像头",
  SecurityError: "摄像头已被禁用，请在系统设置或者浏览器设置中开启后重试",
  TypeError: "类型错误，未检测到可用摄像头"
};

class FaceDetection {
  constructor(options) {
    this.options = Object.assign({
        matchedScore: 0.7,
        mediaSize: {
          width: 540,
          height: 325
        }
      },
      options
    );

    this.timer = null;
    this.mediaStreamTrack = null; // 摄像头媒体流
    this.descriptors = {
      desc1: null,
      desc2: null
    }; // 两种数据

    this.videoEl = document.querySelector("#videoEl"); // 视频区域
    this.trackBoxEl = document.querySelector("#trackBox"); // 人脸框绘制区域
    this.canvasImgEl = document.querySelector("#canvasImg"); // 图片绘制区域

    this.init();
  }

  async init() {
    this.resize();
    await this.initDetection();
    this.loadFaceImage();
  }

  // 设置相关容器大小
  resize() {
    const tmp = [this.videoEl, this.canvasImgEl];
    for (let i = 0; i < tmp.length; i++) {
      tmp[i].width = this.options.mediaSize.width;
      tmp[i].height = this.options.mediaSize.height;
    }
    const wraperEl = document.querySelector(".wraper");
    wraperEl.style.width = `${this.options.mediaSize.width}px`;
    wraperEl.style.height = `${this.options.mediaSize.height}px`;
  }

  // 初始化人脸识别
  async initDetection() {
    // 加载模型
    await faceapi.loadTinyFaceDetectorModel("./assets/face-api/weights");
    const mediaOpt = {
      video: true
    };
    // 获取 WebRTC 媒体视频流
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      // 最新标准API
      this.mediaStreamTrack = await navigator.mediaDevices
        .getUserMedia(mediaOpt)
        .catch(this.mediaErrorCallback);
    }
    this.initVideo();
  }

  // 初始化视频流
  initVideo() {
    this.videoEl.onplay = () => {
      this.onPlay();
    };
    this.videoEl.srcObject = this.mediaStreamTrack;
    setTimeout(() => this.onPlay(), 300);
  }

  // 获取媒体流错误处理
  mediaErrorCallback(error) {
    if (errorMap[error.name]) {
      alert(errorMap[error.name]);
    }
  }

  // 循环监听扫描视频流中的人脸特征
  async onPlay() {
    // 判断视频对象是否暂停结束
    if (this.videoEl && (this.videoEl.paused || this.videoEl.ended)) {
      this.timer = setTimeout(() => this.onPlay());
      return;
    }

    // 设置 TinyFaceDetector 模型参数：inputSize 输入视频流大小  scoreThreshold 人脸评分阈值
    const faceDetectionTask = await faceapi.detectSingleFace(
      this.videoEl,
      new faceapi.TinyFaceDetectorOptions({
        inputSize: 512,
        scoreThreshold: 0.6
      })
    );

    // 判断人脸扫描结果
    if (faceDetectionTask) {
      if (faceDetectionTask.score > this.options.matchedScore) {
        console.log(`检测到人脸，匹配度大于 ${this.options.matchedScore}`);
        // 人脸符合要求，暂停视频流
        this.videoEl.pause();
        // TODO 调用后端接口进行身份验证
        this.canvasImgEl
          .getContext("2d")
          .drawImage(
            this.videoEl,
            0,
            0,
            this.canvasImgEl.width,
            this.canvasImgEl.height
          );
        // 将绘制的图像转化成 图片的 base64 编码
        let image = this.canvasImgEl.toDataURL("image/png");
        this.descriptors.desc2 = image;
        this.recognitionFace();
      }
    }
    this.timer = setTimeout(() => this.onPlay());
  }

  recognitionFace() {
    const img1 = document.createElement("img");
    const img2 = document.createElement("img");
    img1.src = this.descriptors.desc1;
    img2.src = this.descriptors.desc2;

    document.body.appendChild(img1);
    document.body.appendChild(img2);

    setTimeout(async () => {
      console.log("img1", img1, "img2", img2);
      const desc1 = await faceapi.computeFaceDescriptor(img1);
      // face-api.js:3558 Uncaught (in promise) Error: FaceRecognitionNet - load model before inference
      // at FaceRecognitionNet../src/assets/face-api/face-api.js.FaceRecognitionNet.forwardInput (face-api.js:3558)
      const desc2 = await faceapi.computeFaceDescriptor(img2);
      console.log("desc1", desc1, "desc2", desc2);

      // desc1 是Float32Array数据
      // todo 报错
      const distance = faceapi.utils.round(
        faceapi.euclideanDistance(desc1, desc2)
      );

      Error: euclideanDistance: arr1.length !== arr2.length;
      console.log("distance", distance);
    }, 3000);
  }

  loadFaceImage() {
    let url = "./assets/images/11.jpg";
    this.convertImgToBase64(url, base64Img => {
      //转化后的base64
      this.descriptors.desc1 = base64Img;
    });
  }

  convertImgToBase64(url, callback) {
    let canvas = document.createElement("CANVAS"),
      ctx = canvas.getContext("2d"),
      img = new Image();
    img.crossOrigin = "Anonymous";
    img.onload = function () {
      canvas.height = img.height;
      canvas.width = img.width;
      ctx.drawImage(img, 0, 0);
      let dataURL = canvas.toDataURL("image/png");
      callback.call(this, dataURL);
    };
    img.src = url;
  }
}

module.exports = FaceDetection;