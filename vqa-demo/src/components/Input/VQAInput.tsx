/* External dependencies */
import { Dispatch, useMemo, useState } from 'react'
import {
  Button,
  Flex,
  FormControl,
  FormLabel,
  Heading,
  Input,
  Stack,
  Container,
  FormErrorMessage,
} from '@chakra-ui/react'
import axios from 'axios'
import { useForm } from 'react-hook-form'
import { Icon } from '@chakra-ui/icons'
import { FiFile } from 'react-icons/fi'
import url from './url.json'

/* Internal dependencies */
import ImageUpload from './ImageUpload'
import Thumbnail from './Thumbnail'
import VqaResultProps from '../../types/VqaResult.types'

type FormValues = {
  file: FileList
  question: string
}

interface VQAInputProps {
  setVqaResult: Dispatch<VqaResultProps>
}

function VQAInputScreen({
  setVqaResult,
}: VQAInputProps) {
  const { register, handleSubmit, formState: { errors } } = useForm<FormValues>()

  const [loading, setLoading] = useState(false)
  const [imageSrc, setImageSrc] = useState('')

  const encodeFileToBase64 = (file: File) => {
    setLoading(true)
    const reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onloadend = () => {
      setLoading(false)
      // @ts-ignore
      setImageSrc(reader.result)
    }
  }

  const validateFiles = (value: FileList) => {
    if (value.length < 1) {
      return 'Files is required'
    }
    for (const file of Array.from(value)) {
      const fsMb = file.size / (1024 * 1024)
      const MAX_FILE_SIZE = 10
      if (fsMb > MAX_FILE_SIZE) {
        return 'Max file size 10mb'
      }
    }
    return true
  }

  const thumbNailComponent = useMemo(() => (
    <Thumbnail
      imageSrc={imageSrc}
      loading={loading}
    />
  ), [
    imageSrc,
    loading
  ])

  // TODO: API 연결하기
  const onSubmit = handleSubmit(async (data) => {

    const base64image = imageSrc.slice(imageSrc.indexOf(',') + 1)
    const { serverUrl } = url
    const res = (await axios.post(serverUrl, {
      base64image,
      question: data.question,
    })).data
    setVqaResult({
      MCAoAN: {
        oriQuestion: res.ori_question,
        oriImage: res.ori_image,
        questionData: res.question_data,
        boxedImage: res.boxed_image,
        importantBoxes: res.important_boxes,
        answer: res.answer,
      },
      LSTM: {
        oriQuestion: res.ori_question,
        oriImage: res.ori_image,
        questionData: res.question_data,
        boxedImage: res.boxed_image,
        importantBoxes: res.important_boxes,
        answer: 'LSTM Test',
      },
      SBERT: {
        oriQuestion: res.ori_question,
        oriImage: res.ori_image,
        questionData: res.question_data,
        boxedImage: res.boxed_image,
        importantBoxes: res.important_boxes,
        answer: 'SBERT Test',
      }
    })
  })

  return (
    <Container>
      <Stack minH={'30vh'} direction={{ base: 'column', md: 'row' }}>
        <Flex flex={1} direction="column">
          <Heading fontSize={'3xl'} mb={8}>
            Try VQA on your Image
          </Heading>
          <form onSubmit={onSubmit}>
            <Stack spacing={4} w={'full'} maxW={'md'}>
              <FormControl id="image" isInvalid={!!errors.file} isRequired>
                <FormLabel>
                  Image
                </FormLabel>
                <ImageUpload
                  accept={'image/*'}
                  register={register('file', {
                    validate: validateFiles,
                    onChange: e => { encodeFileToBase64(e.target.files[0]) },
                  })}
                >
                  <Stack spacing={4} w={'full'}>
                    { thumbNailComponent }
                    <Button leftIcon={<Icon as={FiFile} />}>
                      Upload
                    </Button>
                  </Stack>
                </ImageUpload>

                <FormErrorMessage>
                  {errors.file && errors?.file.message}
                </FormErrorMessage>
              </FormControl>

              <FormControl id="question" isRequired>
                <FormLabel>
                  Question
                </FormLabel>
                <Input
                  placeholder='write the Question'
                  {...register('question')}
                />
              </FormControl>
              <Button
                type='submit'
                colorScheme='red'
                bg='red.400'
                _hover={{ bg: 'red.500' }}
                variant='solid'
              >
                Submit
              </Button>
            </Stack>
          </form>
        </Flex>
      </Stack>
    </Container>
  )
}

export default VQAInputScreen
