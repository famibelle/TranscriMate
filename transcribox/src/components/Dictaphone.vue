<!-- Dictaphone.vue -->
<template>
  <div class="dictaphone">
    <!-- Timer -->
    <!-- <div class="timer" :class="{ 'recording': isRecording }">
      {{ formatTime(recordingTime) }}
    </div> -->

    <!-- Waveform visualization -->
    <!-- <div class="waveform" :class="{ 'recording': isRecording }">
      <template v-for="i in 7" :key="i">
        {{ getWaveformBar(i) }}
      </template>
    </div> -->

    <!-- Controls -->
    <div class="controls">
      <button class="control-btn" @click="onStop" v-if="isRecording">
        ⬛️ Stop
      </button>
      <button class="control-btn" @click="onRecord" v-else>
        🔴 Enregistrer
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Dictaphone',
  props: {
    isRecording: {
      type: Boolean,
      default: false
    },
    audioLevel: {
      type: Number,
      default: 0
    }
  },
  data() {
    return {
      recordingTime: 0,
      waveformBars: ['▁', '▂', '▃', '▄', '▅', '▆', '▇'],
      timer: null
    }
  },
  methods: {
    formatTime(seconds) {
      const minutes = Math.floor(seconds / 60)
      const remainingSeconds = seconds % 60
      return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
    },
    getWaveformBar(index) {
      if (!this.isRecording) return '▁'
      
      // Créer un effet d'onde animée
      const time = Date.now() / 200
      const amplitude = this.audioLevel * 0.7 + 0.3 // Minimum 0.3, maximum 1.0
      const wave = Math.sin(time + index) * amplitude
      const barIndex = Math.floor((wave + 1) * (this.waveformBars.length - 1) / 2)
      return this.waveformBars[barIndex]
    },
    onRecord() {
      this.$emit('record')
      this.startTimer()
    },
    onStop() {
      this.$emit('stop')
      this.stopTimer()
    },
    startTimer() {
      this.timer = setInterval(() => {
        this.recordingTime++
      }, 1000)
    },
    stopTimer() {
      if (this.timer) {
        clearInterval(this.timer)
        this.timer = null
      }
      this.recordingTime = 0
    }
  },
  beforeUnmount() {
    this.stopTimer()
  }
}
</script>

<style scoped>
.dictaphone {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
  padding: 20px;
  background: #f5f5f5;
  border-radius: 12px;
  min-width: 300px;
}

.timer {
  font-size: 2em;
  font-weight: 300;
  color: #333;
}

.timer.recording {
  color: #ff3b30;
}

.waveform {
  font-family: monospace;
  font-size: 24px;
  letter-spacing: 2px;
  color: #666;
  height: 40px;
  display: flex;
  align-items: center;
  gap: 2px;
}

.waveform.recording {
  color: #ff3b30;
}

.controls {
  margin-top: 20px;
}

.control-btn {
  font-size: 1.2em;
  padding: 10px 20px;
  border: none;
  background: transparent;
  cursor: pointer;
  border-radius: 20px;
  transition: all 0.3s ease;
}

.control-btn:hover {
  background: rgba(0, 0, 0, 0.05);
}
</style>