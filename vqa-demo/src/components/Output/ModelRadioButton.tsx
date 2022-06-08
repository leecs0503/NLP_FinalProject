import { Dispatch, useState } from 'react'
import { Box, HStack, useRadio, useRadioGroup } from '@chakra-ui/react'

import { ModelName } from '../../constant/modelName'

// @ts-ignore
function RadioCard(props) {
  const { getInputProps, getCheckboxProps } = useRadio(props)

  const input = getInputProps()
  const checkbox = getCheckboxProps()

  return (
    <Box as='label'>
      <input {...input} />
      <Box
        {...checkbox}
        cursor='pointer'
        borderWidth='1px'
        borderRadius='md'
        boxShadow='md'
        _checked={{
          bg: 'red.400',
          color: 'white',
          borderColor: 'red.400',
        }}
        _focus={{
          boxShadow: 'outline',
        }}
        px={5}
        py={3}
      >
        {props.children}
      </Box>
    </Box>
  )
}

interface ModelRadioProps {
  currentModel: ModelName,
  setCurrentModel: Dispatch<ModelName>,
}

function ModelRadioButton({
  currentModel,
  setCurrentModel,
}: ModelRadioProps) {

  const modelNameArray = Object.values(ModelName)

  const handleChangeModel = (value: ModelName) => {
    setCurrentModel(value)
  }

  const { getRootProps, getRadioProps } = useRadioGroup({
    name: 'modelName',
    defaultValue: currentModel,
    onChange: handleChangeModel,
  })

  const group = getRootProps()

  return (
    <HStack {...group}>
      {modelNameArray.map((value) => {
        const radio = getRadioProps({ value })
        return (
          <RadioCard key={value} {...radio}>
            {value}
          </RadioCard>
        )
      })}
    </HStack>
  )
}

export default ModelRadioButton
