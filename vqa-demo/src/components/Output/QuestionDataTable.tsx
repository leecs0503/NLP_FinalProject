import { HStack, Text, VStack } from '@chakra-ui/react'
import { QuestionDataProps, VqaOutPutProps } from '../../types/VqaResult.types'

type QuestionDataTableType = Pick<VqaOutPutProps, 'questionData'>

function QuestionDataTable({
  questionData,
}: QuestionDataTableType) {
  const wordCount = questionData.length

  const getTextStyle = (_wordCount: number, att: number) => {
    if (att > 2/_wordCount) {
      return {
        color: 'red.500',
        fontWeight: 'medium',
      }
    }
    else if (att > 1/_wordCount) {
      return {
        color: 'gray.700',
        fontWeight: 'normal',
      }
    }
    else {
      return {
        color: 'gray.400',
        fontWeight: 'normal',
      }
    }
  }

  return (
    <HStack p={3} spacing={3}>
      <VStack align='flex-end' mr={4}>
        <Text fontSize='lg' fontWeight='medium'>Word</Text>
        <Text fontSize='lg' fontWeight='medium'>Att</Text>
      </VStack>
      { questionData.map((x: QuestionDataProps) => {
        const textStyle = getTextStyle(wordCount, x.att)
        return (
          <VStack>
            <Text fontSize='lg' color={textStyle.color} fontWeight={textStyle.fontWeight}>
              {x.word}
            </Text>
            <Text fontSize='lg' color={textStyle.color} fontWeight={textStyle.fontWeight}>
              {x.att.toFixed(2)}
            </Text>
          </VStack>
        )
      })}
    </HStack>
  )
}

export default QuestionDataTable
