import {
  Box,
  Container,
  Divider,
  Heading,
  Text,
  VStack,
  Image,
  Grid,
  GridItem,
} from '@chakra-ui/react'
import { useMemo, useState } from 'react'
import VqaResultProps, { ImportantBoxProps } from '../../types/VqaResult.types'
import { WarningTwoIcon } from '@chakra-ui/icons'
import ModelRadioButton from './ModelRadioButton'
import QuestionDataTable from './QuestionDataTable'
import { ModelName } from '../../constant/modelName'

const Base64toImg = (
  imageData: string,
  boxSize?: string,
) => (
  <Image boxSize={boxSize} src={`data:image/jpeg;base64,${imageData}`} />
)

function VQAOutPutScreen({
  MCAoAN,
  LSTM,
  SBERT,
}: VqaResultProps) {
  const [currentModel, setCurrentModel] = useState(ModelName.MCAoAN)

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const modelDataKeyMap = {
    [ModelName.MCAoAN]: MCAoAN,
    [ModelName.LSTM]: LSTM,
    [ModelName.SBERT]: SBERT,
  }

  const ResultComponent = useMemo(() => (
    <VStack
      divider={
        <Divider />
      }
    >
      <Box
        overflow='hidden'
        flex='stretch'
        p={4}
      >
        <VStack spacing={4} divider={<Divider />} align='stretch'>
          <Box mb={4}>
            <Heading fontSize='2xl' mb={6}>Select Output Model</Heading>
            <ModelRadioButton
              currentModel={currentModel}
              setCurrentModel={setCurrentModel}
            />
          </Box>

          <Box>
            <Heading fontSize='xl' mb={2}>Question</Heading>
            <Text fontSize='lg'>{modelDataKeyMap[currentModel].oriQuestion}</Text>
          </Box>

          <Box>
            <Heading fontSize='xl' mb={2}>Answer</Heading>
            <Text fontSize='lg'>{modelDataKeyMap[currentModel].answer}</Text>
          </Box>

          <Box>
            <Heading fontSize='xl' mb={2}>Original Image</Heading>
            {
              modelDataKeyMap[currentModel].oriImage !== "" && (
                Base64toImg(modelDataKeyMap[currentModel].oriImage)
              )
            }
          </Box>

          <Box>
            <Heading fontSize='xl' mb={4}>Question Word Data</Heading>
            <QuestionDataTable questionData={modelDataKeyMap[currentModel].questionData} />
          </Box>

          <Box>
            <Heading fontSize='xl' mb={2}>
              Image with attention box
            </Heading>
            { modelDataKeyMap[currentModel].boxedImage !== "" && (
              Base64toImg(modelDataKeyMap[currentModel].boxedImage)
            )}
          </Box>

          <Box>
            <Heading fontSize='xl' mb={2}>
              Important Attention box list
            </Heading>
            <Text fontSize='sm' color='gray.500' mb={4}>
              Attention box list which has attention weight more than avg
            </Text>
            <Grid templateColumns='repeat(3, 2fr)' gap={6}>
              {
                modelDataKeyMap[currentModel].importantBoxes.map((obj: ImportantBoxProps) => (
                  <GridItem alignContent='center'>
                    { Base64toImg(obj.str, '200px') }
                    <Text fontSize='md' color='gray.600' align='center'>
                      {'attention: '}{ obj.att.toFixed(2) }
                    </Text>
                  </GridItem>
                ))
              }
            </Grid>
          </Box>
        </VStack>
      </Box>
    </VStack>
  ), [currentModel, modelDataKeyMap])

  const EmptyInputComponent = useMemo(() => (
    <Box textAlign='center' py={10} px={6}>
      <WarningTwoIcon boxSize='50px' color='orange.300' />
      <Heading size='lg' mt={6} mb={2}>
        Please input Image and Question!
      </Heading>
    </Box>
  ), [])

  return (
    <Container>
      <Heading fontSize='3xl' mb={8}>
        VQA Result
      </Heading>
      { MCAoAN.answer === ''
        ? EmptyInputComponent
        : ResultComponent
      }
    </Container>
  )
}

export default VQAOutPutScreen
