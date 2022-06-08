/* External dependencies */
import * as React from "react"
import { useMediaQuery } from 'react-responsive'
import {
  ChakraProvider,
  HStack,
  StackDivider
} from "@chakra-ui/react"

/* Internal dependencies */
import Header from './components/Header'
import theme from './theme'
import Hero from './components/Hero'
import VQAInputScreen from './components/Input/VQAInput'
import VQAOutPutScreen from './components/Output/VQAOutPut'
import VqaResultProps, { VqaOutPutProps } from './types/VqaResult.types'

const VqaOutPutInitialState: VqaOutPutProps = {
  oriQuestion: '',
  oriImage: '',
  questionData: [],
  boxedImage: '',
  importantBoxes: [],
  answer: '',
}

const VqaResultInitialState : VqaResultProps = {
  MCAoAN: VqaOutPutInitialState,
  LSTM: VqaOutPutInitialState,
  SBERT: VqaOutPutInitialState,
}

export const App = () => {
  const [vqaResult, setVqaResult] = React.useState(VqaResultInitialState)
  
  const isDesktopOrLaptop = useMediaQuery({
    query: '(min-width: 520px)'
  })
  return (
    <ChakraProvider theme={theme}>
      <Header />
      <Hero />
      {
        isDesktopOrLaptop && <HStack
          divider={<StackDivider borderColor='gray.200'/>}
          spacing={4}
          alignItems='flex-start'
          mx='118px'
          mb={8}
        >
          <VQAInputScreen
            setVqaResult={setVqaResult}
          />
          <VQAOutPutScreen
            MCAoAN={vqaResult.MCAoAN}
            LSTM={vqaResult.LSTM}
            SBERT={vqaResult.SBERT}
          />
        </HStack>
      }
      {
        !isDesktopOrLaptop && 
        <React.Fragment>
          <VQAInputScreen
            setVqaResult={setVqaResult}
          />
          <VQAOutPutScreen
            MCAoAN={vqaResult.MCAoAN}
            LSTM={vqaResult.LSTM}
            SBERT={vqaResult.SBERT}
          />
        </React.Fragment>
      }
      
    </ChakraProvider>
  )
}
