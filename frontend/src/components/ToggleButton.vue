<template>
  <div class="toggle-container">
    <label class="toggle">
      <input
        type="checkbox"
        :checked="modelValue"
        @change="updateValue"
      >
      <span class="toggle-slider">
        <span class="toggle-button"></span>
      </span>
    </label>
    <span v-if="label" class="toggle-label">{{ label }}</span>
  </div>
</template>

<script>
export default {
  name: 'ToggleButton',
  props: {
    modelValue: {
      type: Boolean,
      required: true
    },
    label: {
      type: String,
      default: ''
    }
  },
  emits: ['update:modelValue'],
  setup(props, { emit }) {
    const updateValue = (event) => {
      emit('update:modelValue', event.target.checked)
    }

    return {
      updateValue
    }
  }
}
</script>

<style scoped>
.toggle-container {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.toggle {
  position: relative;
  display: inline-block;
  width: 50px;
  height: 26px;
  cursor: pointer;
}

.toggle input {
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
  background-color: #e0e0e0;
  border-radius: 34px;
  transition: background-color 0.3s;
}

.toggle-button {
  position: absolute;
  content: "";
  height: 20px;
  width: 20px;
  left: 3px;
  top: 3px;
  background-color: white;
  border-radius: 50%;
  transition: transform 0.3s;
  box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

/* État actif */
input:checked + .toggle-slider {
  background-color: #4CAF50;
}

input:checked + .toggle-slider .toggle-button {
  transform: translateX(24px);
}

/* Focus state */
input:focus + .toggle-slider {
  box-shadow: 0 0 1px #4CAF50;
}

.toggle-label {
  font-size: 14px;
  user-select: none;
}
</style>