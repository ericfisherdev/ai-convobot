import { Button } from "../../components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogTrigger,
} from "../../components/ui/dialog"

import {
    Drawer,
    DrawerClose,
    DrawerContent,
    DrawerDescription,
    DrawerFooter,
    DrawerHeader,
    DrawerTrigger,
  } from "../../components/ui/drawer"
import { Settings } from "lucide-react"
import { EditData } from "./EditData"
import { useMobile } from "../../hooks/useMobile"

export function EditDataPopup() {
    const { isMobile, isTablet } = useMobile();
    const useMobileLayout = isMobile || isTablet;
  return (
    <>
    {useMobileLayout ? 
    <Drawer>
    <DrawerTrigger asChild>
      <Button variant="outline" size={"sm"} className="touch-target">
        <Settings className="h-4 w-4" />
      </Button>
    </DrawerTrigger>
    <DrawerContent className="max-h-[85vh]">
      <DrawerHeader>
      </DrawerHeader>
      <DrawerDescription>
        <EditData />
      </DrawerDescription>
      <DrawerFooter>
        <DrawerClose>
          <Button variant="outline">Cancel</Button>
        </DrawerClose>
      </DrawerFooter>
    </DrawerContent>
  </Drawer>
    :
    <Dialog>
    <DialogTrigger asChild>
      <Button variant="outline" size={"sm"}>
        <Settings className="h-4 w-4" />
      </Button>
    </DialogTrigger>
    <DialogContent className="sm:max-w-[650px] max-h-[85vh] overflow-hidden">
      <EditData />
    </DialogContent>
  </Dialog>
    }
    </>
  )
}
