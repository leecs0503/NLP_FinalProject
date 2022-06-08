import { ReactNode, useRef } from 'react'
import { Input, InputGroup } from '@chakra-ui/react'
import { UseFormRegisterReturn } from 'react-hook-form'

type FileUploadProps = {
  register: UseFormRegisterReturn
  accept?: string
  children?: ReactNode
}

function ImageUpload(props: FileUploadProps){
  const { register, accept, children } = props
  const inputRef = useRef<HTMLInputElement | null>(null)
  const { ref, ...rest } = register as {ref: (instance: HTMLInputElement | null) => void}

  const handleClick = () => inputRef.current?.click()

  return (
    <InputGroup onClick={handleClick}>
      <Input
        type={'file'}
        hidden
        multiple={false}
        accept={accept}
        {...rest}
        ref={(e) => {
          ref(e)
          inputRef.current = e
        }}
      />
      <>
        {children}
      </>
    </InputGroup>
  )
}

export default ImageUpload
