<template>
  <div class="custom-toggle-container">
    <label class="custom-toggle" :style="{ width: width + 'px', height: height + 'px' }">
      <input
        type="checkbox"
        :checked="isRightOptionSelected"
        @change="updateValue"
      >
      <div class="toggle-slider" :style="sliderStyles">
        <div class="label-container">
          <span 
            class="toggle-text" 
            :class="{ active: !isRightOptionSelected }"
          >
            {{ leftOption.label }}
          </span>
        </div>
        
        <div 
          class="toggle-button" 
          :style="buttonStyles"
        ></div>
        
        <div class="label-container">
          <span 
            class="toggle-text" 
            :class="{ active: isRightOptionSelected }"
          >
            {{ rightOption.label }}
          </span>
        </div>
      </div>
    </label>
  </div>
</template>

<script>
export default {
  name: 'CustomToggle',
  props: {
    modelValue: {
      required: true
    },
    leftOption: {
      type: Object,
      required: true,
      validator: (obj) => 'value' in obj && 'label' in obj
    },
    rightOption: {
      type: Object,
      required: true,
      validator: (obj) => 'value' in obj && 'label' in obj
    },
    width: {
      type: Number,
      default: 250
    },
    height: {
      type: Number,
      default: 40
    }
  },
  
  emits: ['update:modelValue'],
  
  computed: {
    isRightOptionSelected() {
      return this.modelValue === this.rightOption.value
    },
    buttonStyles() {
      const buttonWidth = this.width / 2
      return {
        width: `${buttonWidth}px`,
        transform: this.isRightOptionSelected ? `translateX(${buttonWidth}px)` : 'translateX(0)'
      }
    },
    sliderStyles() {
      return {
        backgroundColor: '#f0f0f0',
        borderRadius: (this.height / 2) + 'px',
        height: this.height + 'px'
      }
    }
  },
  
  methods: {
    updateValue(event) {
      const newValue = event.target.checked ? this.rightOption.value : this.leftOption.value
      this.$emit('update:modelValue', newValue)
    }
  }
}
</script>

<style scoped>
.custom-toggle-container {
  display: inline-block;
}

.custom-toggle {
  position: relative;
  display: inline-block;
  cursor: pointer;
}

.custom-toggle input {
  display: none;
}

.toggle-slider {
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: relative;
  overflow: hidden; /* Pour éviter tout débordement */
}

.label-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  position: relative;
  z-index: 2;
}

.toggle-button {
  position: absolute;
  height: 100%; /* Changé pour occuper toute la hauteur */
  background-color: white;
  border-radius: inherit; /* Hérite du borderRadius parent */
  transition: transform 0.3s ease;
  box-shadow: 0 2px 4px rgba(0,0,0,0.2);
  top: 0; /* Aligné au top */
  left: 0;
  z-index: 1;
}

.toggle-text {
  font-size: 14px;
  color: #666;
  transition: color 0.3s;
  user-select: none;
  padding: 0 8px;
}

.toggle-text.active {
  color: #2196F3;
  font-weight: 500;
}

input:focus + .toggle-slider {
  box-shadow: 0 0 1px #2196F3;
}
</style>