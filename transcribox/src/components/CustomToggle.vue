<template>
    <div class="custom-toggle-container">
      <label class="custom-toggle" :style="{ width: width + 'px', height: height + 'px' }">
        <input
          type="checkbox"
          :checked="isRightOptionSelected"
          @change="updateValue"
        >
        <span class="toggle-slider" :style="sliderStyles">
          <div class="toggle-content">
            <span 
              class="toggle-text left" 
              :class="{ active: !isRightOptionSelected }"
              :style="textStyles"
            >
              {{ leftOption.label }}
            </span>
            <span 
              class="toggle-text right" 
              :class="{ active: isRightOptionSelected }"
              :style="textStyles"
            >
              {{ rightOption.label }}
            </span>
            <span 
              class="toggle-button" 
              :style="buttonStyles"
            ></span>
          </div>
        </span>
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
        default: 200
      },
      height: {
        type: Number,
        default: 40
      },
      activeColor: {
        type: String,
        default: '#2196F3'
      },
      backgroundColor: {
        type: String,
        default: '#f0f0f0'
      },
      textColor: {
        type: String,
        default: '#666'
      }
    },
    
    emits: ['update:modelValue'],
    
    computed: {
      isRightOptionSelected() {
        return this.modelValue === this.rightOption.value
      },
      buttonStyles() {
        const buttonWidth = (this.width - 6) / 2
        const buttonHeight = this.height - 6
        const translation = this.width - buttonWidth - 6
        
        return {
          height: buttonHeight + 'px',
          width: buttonWidth + 'px',
          transform: this.isRightOptionSelected ? `translateX(${translation}px)` : 'translateX(3px)'
        }
      },
      sliderStyles() {
        return {
          backgroundColor: this.backgroundColor,
          borderRadius: (this.height / 2) + 'px'
        }
      },
      textStyles() {
        return {
          fontSize: Math.max((this.height / 3), 14) + 'px',
          '--active-color': this.activeColor,
          color: this.textColor
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
    margin: 10px 0;
  }
  
  .custom-toggle {
    position: relative;
    display: inline-block;
    cursor: pointer;
  }
  
  .custom-toggle input {
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
    transition: all 0.3s;
  }
  
  .toggle-content {
    position: relative;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 15px;
  }
  
  .toggle-button {
    position: absolute;
    top: 3px;
    background-color: white;
    border-radius: 17px;
    transition: transform 0.3s;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    z-index: 1;
  }
  
  .toggle-text {
    flex: 1;
    font-weight: 500;
    text-align: center;
    transition: color 0.3s;
    user-select: none;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    z-index: 2;
  }
  
  .toggle-text.left {
    padding-right: 10px;
  }
  
  .toggle-text.right {
    padding-left: 10px;
  }
  
  .toggle-text.active {
    color: var(--active-color);
  }
  
  input:focus + .toggle-slider {
    box-shadow: 0 0 1px var(--active-color);
  }
  
  /* Ajustement pour la visibilité du texte */
  .toggle-text {
    mix-blend-mode: difference;
  }
  </style>