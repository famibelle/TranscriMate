<template>
  <div class="task-toggle-container">
    <label class="task-toggle">
      <input
        type="checkbox"
        :checked="modelValue === 'translate'"
        @change="updateValue"
      >
      <span class="toggle-slider">
        <span class="toggle-text left" :class="{ active: modelValue === 'transcribe' }">
          Transcribe
        </span>
        <span class="toggle-button"></span>
        <span class="toggle-text right" :class="{ active: modelValue === 'translate' }">
          Translate
        </span>
      </span>
    </label>
  </div>
</template>

<script>
export default {
  name: 'TaskToggle',
  props: {
    modelValue: {
      type: String,
      required: true,
      validator: (value) => ['transcribe', 'translate'].includes(value)
    }
  },
  emits: ['update:modelValue'],
  setup(props, { emit }) {
    const updateValue = (event) => {
      emit('update:modelValue', event.target.checked ? 'translate' : 'transcribe')
    }

    return {
      updateValue
    }
  }
}
</script>

<style scoped>
.task-toggle-container {
  display: inline-block;
  margin: 10px 0;
}

.task-toggle {
  position: relative;
  display: inline-block;
  width: 200px;
  height: 40px;
  cursor: pointer;
}

.task-toggle input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-slider {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #f0f0f0;
  border-radius: 20px;
  transition: all 0.3s;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 10px;
}

.toggle-button {
  position: absolute;
  height: 34px;
  width: 95px;
  left: 3px;
  top: 3px;
  background-color: white;
  border-radius: 17px;
  transition: transform 0.3s;
  box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

/* État actif */
input:checked + .toggle-slider .toggle-button {
  transform: translateX(99px);
}

.toggle-text {
  z-index: 1;
  font-size: 14px;
  font-weight: 500;
  color: #666;
  transition: color 0.3s;
  user-select: none;
}

.toggle-text.active {
  color: #2196F3;
}

/* Focus state */
input:focus + .toggle-slider {
  box-shadow: 0 0 1px #2196F3;
}
</style>